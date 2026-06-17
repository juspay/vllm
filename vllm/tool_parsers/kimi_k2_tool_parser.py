# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any, NoReturn

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import partial_tag_overlap


def _raise_malformed_tool_call(reason: str) -> NoReturn:
    """Raise serving.MalformedToolCallError, lazily imported to break the
    import cycle (serving.py imports this module). Annotated NoReturn so
    callers get correct type narrowing after the call."""
    from vllm.entrypoints.openai.chat_completion.serving import (
        MalformedToolCallError,
    )

    raise MalformedToolCallError(reason)


try:
    from json_repair import repair_json as _repair_json

    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False
    _repair_json = None  # type: ignore[assignment]

logger = init_logger(__name__)

if not _HAS_JSON_REPAIR:
    logger.warning(
        "json_repair not installed; malformed Kimi K2 tool-call JSON will not "
        "be auto-repaired. Install with `pip install json-repair`."
    )


# Heuristic key extraction from a malformed JSON string. Used only for
# telemetry to compare keys present before vs after json-repair.
_JSON_KEY_PATTERN = re.compile(r'"([^"\\]+)"\s*:')


def _all_keys_in_value(obj: object) -> set[str]:
    """Collect all dict keys at every nesting level of a parsed JSON value."""
    if isinstance(obj, dict):
        out = set(obj.keys())
        for v in obj.values():
            out |= _all_keys_in_value(v)
        return out
    if isinstance(obj, list):
        out: set[str] = set()
        for v in obj:
            out |= _all_keys_in_value(v)
        return out
    return set()


def _structural_diff(orig_str: str, repaired_str: str) -> str:
    """Best-effort structural diff between malformed orig and repaired JSON,
    so operators can spot when json-repair drops required fields."""
    try:
        orig_keys = set(_JSON_KEY_PATTERN.findall(orig_str))
        repaired_keys = _all_keys_in_value(json.loads(repaired_str))
        dropped = sorted(orig_keys - repaired_keys)
        added = sorted(repaired_keys - orig_keys)
        return f"keys_dropped={dropped} keys_added={added}"
    except Exception:
        return "diff_unavailable"


def _validate_or_repair_args(
    args_str: str,
    fn_name: str,
    request_id: str | None = None,
) -> tuple[str | None, Any, bool, str | None]:
    """Strict JSON parse with json-repair fallback.

    Returns ``(args, parsed, was_repaired, orig_err)``:
      * ``(args_str, parsed, False, None)`` - input parses strictly.
      * ``(repaired, parsed, True, orig_err)`` - json-repair recovered it.
      * ``(None, None, True, orig_err)`` - unrecoverable.
    """
    try:
        parsed = json.loads(args_str)
        return args_str, parsed, False, None
    except json.JSONDecodeError as e:
        orig_err = f"{e.msg} at pos {e.pos}"
        logger.warning(
            "Malformed Kimi K2 tool args for %s (request_id=%s) at pos %d: "
            "%s. arg_len=%d. Attempting repair.",
            fn_name,
            request_id,
            e.pos,
            e.msg,
            len(args_str),
        )
    if not _HAS_JSON_REPAIR or _repair_json is None:
        return None, None, True, orig_err
    try:
        repaired = _repair_json(args_str)
        if not repaired:
            return None, None, True, orig_err
        parsed = json.loads(repaired)
    except Exception as err:
        logger.error("json_repair could not recover args for %s: %s", fn_name, err)
        return None, None, True, orig_err
    logger.warning(
        "Repaired Kimi K2 tool args for %s (request_id=%s) len %d -> %d. %s",
        fn_name,
        request_id,
        len(args_str),
        len(repaired),
        _structural_diff(args_str, repaired),
    )
    return repaired, parsed, True, orig_err


class KimiK2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        # Streaming state
        self._sent_content_idx: int = 0
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        # Indices of streamed tool calls whose final args have been
        # validated/repaired at close, to avoid re-validating each delta.
        self._validated_tool_idx: set[int] = set()

        # Section marker
        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"

        # Individual tool call markers
        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.tool_call_arg_token: str = "<|tool_call_argument_begin|>"

        # Regex for non-streaming extraction
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*"
            r"<\|tool_call_argument_begin\|>\s*"
            r"(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*"
            r"<\|tool_call_end\|>",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure special-token markers appear as literal text in
            # current_text so we can do pure text-based parsing.
            request.skip_special_tokens = False
        return request

    @staticmethod
    def _extract_function_name(tool_call_id: str) -> str:
        """Extract function name from a Kimi K2 tool-call id.

        'functions.Read:0' -> 'Read', 'get_weather:0' -> 'get_weather'.
        """
        if ":" in tool_call_id:
            return tool_call_id.split(":")[0].split(".")[-1]
        if "." in tool_call_id:
            return tool_call_id.split(".")[-1]
        return tool_call_id

    def _get_tool_schema(self, function_name: str) -> dict[str, Any] | None:
        for tool in self.tools or []:
            function = getattr(tool, "function", None)
            if function is None and isinstance(tool, dict):
                function = tool.get("function")

            name = getattr(function, "name", None)
            if name is None and isinstance(function, dict):
                name = function.get("name")

            if name != function_name:
                continue

            parameters = getattr(function, "parameters", None)
            if parameters is None and isinstance(function, dict):
                parameters = function.get("parameters")
            return parameters if isinstance(parameters, dict) else {}
        return None

    @staticmethod
    def _json_type_name(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        if value is None:
            return "null"
        return type(value).__name__

    @classmethod
    def _matches_json_type(cls, value: Any, expected: str) -> bool:
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "string":
            return isinstance(value, str)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "null":
            return value is None
        return True

    @classmethod
    def _validate_schema_subset(
        cls,
        value: Any,
        schema: dict[str, Any],
        path: str = "$",
    ) -> list[str]:
        errors: list[str] = []

        expected_type = schema.get("type")
        expected_types = (
            expected_type if isinstance(expected_type, list) else [expected_type]
        )
        expected_types = [t for t in expected_types if isinstance(t, str)]
        if expected_types and not any(
            cls._matches_json_type(value, expected) for expected in expected_types
        ):
            errors.append(
                f"{path} expected {'/'.join(expected_types)} got "
                f"{cls._json_type_name(value)}"
            )
            return errors

        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            errors.append(f"{path} value {value!r} not in enum")

        if not isinstance(value, dict):
            return errors

        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    errors.append(f"{path}.{key} is required but missing")

        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key not in value or not isinstance(subschema, dict):
                    continue
                errors.extend(
                    cls._validate_schema_subset(value[key], subschema, f"{path}.{key}")
                )

        return errors

    def _check_tool_call_issue(
        self,
        function_args: str,
        function_name: str,
        parsed_args: Any = None,
    ) -> str | None:
        """Return the malformation reason for a tool call, or None if OK.

        Pass ``parsed_args`` to skip a redundant json.loads. Skips schema/name
        checks when the parser was constructed without a tools list, so unknown
        tool names with valid JSON are NOT gated (only the post-repair path
        reaches here).
        """
        if parsed_args is None:
            try:
                parsed_args = json.loads(function_args)
            except json.JSONDecodeError as e:
                return f"invalid JSON arguments: {e}"
        if not self.tools:
            return None
        schema = self._get_tool_schema(function_name)
        if schema is None:
            return "unknown tool name"
        schema_errors = self._validate_schema_subset(parsed_args, schema)
        if schema_errors:
            return "schema mismatch: " + "; ".join(schema_errors)
        return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        else:
            try:
                # there are two possible captures - between tags, or between a
                # tag and end-of-string so the result of
                # findall is an array of tuples where one is a function call and
                # the other is None
                function_call_tuples = self.tool_call_regex.findall(model_output)

                logger.debug("function_call_tuples: %s", function_call_tuples)

                request_id = getattr(request, "request_id", None)
                tool_calls = []
                for match in function_call_tuples:
                    function_id, function_args = match
                    # function_id: functions.get_weather:0 or get_weather:0
                    function_name = self._extract_function_name(function_id)
                    (
                        repaired_args,
                        parsed_args,
                        was_repaired,
                        orig_err,
                    ) = _validate_or_repair_args(
                        function_args, function_name, request_id
                    )
                    if repaired_args is None:
                        # Unrecoverable malformed JSON. Raise so serving.py
                        # returns a 500 and LiteLLM retries; dropping silently
                        # would leave the client with empty tool_calls and no
                        # retry signal.
                        _raise_malformed_tool_call(
                            f"unrecoverable JSON args for {function_name} "
                            f"(tool_call_id={function_id}, "
                            f"request_id={request_id}): {orig_err}"
                        )
                    if was_repaired:
                        # json-repair can produce parseable-but-wrong JSON
                        # (e.g. drop a required field). Validate the repaired
                        # args against the tool schema before emitting.
                        issue = self._check_tool_call_issue(
                            repaired_args, function_name, parsed_args
                        )
                        if issue:
                            _raise_malformed_tool_call(
                                f"post-repair {issue} for {function_name} "
                                f"(tool_call_id={function_id}, "
                                f"request_id={request_id}, orig_err={orig_err})"
                            )
                    tool_calls.append(
                        ToolCall(
                            # Keep nightly's history-indexed id captured from
                            # the model output; do NOT regenerate a random id.
                            id=function_id,
                            type="function",
                            function=FunctionCall(
                                name=function_name, arguments=repaired_args
                            ),
                        )
                    )

                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception as exc:
                # Let MalformedToolCallError propagate so serving.py converts
                # it to a 500 and LiteLLM retries.
                from vllm.entrypoints.openai.chat_completion.serving import (
                    MalformedToolCallError,
                )

                if isinstance(exc, MalformedToolCallError):
                    raise
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent content before the tool-calls section, or None.

        Holds back any trailing suffix that partially matches
        ``<|tool_calls_section_begin|>`` to avoid leaking marker bytes.
        """
        if self.tool_calls_start_token not in current_text:
            overlap = partial_tag_overlap(current_text, self.tool_calls_start_token)
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = current_text.index(self.tool_calls_start_token)

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_calls(self, current_text: str) -> list[tuple[str, bool]]:
        """Extract ``(body, is_closed)`` from
        ``<|tool_call_begin|>…<|tool_call_end|>`` blocks.

        ``is_closed`` is True when the closing ``<|tool_call_end|>`` was seen,
        meaning the body holds the call's final arguments and is safe to
        validate/repair.
        """
        if self.tool_calls_start_token not in current_text:
            return []

        results: list[tuple[str, bool]] = []
        pos = current_text.index(self.tool_calls_start_token)
        while True:
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            tc_start = start + len(self.tool_call_start_token)
            end = current_text.find(self.tool_call_end_token, tc_start)

            if end != -1:
                tool_call = current_text[tc_start:end]
                pos = end + len(self.tool_call_end_token)
                is_closed = True
            else:
                tool_call = current_text[tc_start:]
                overlap = partial_tag_overlap(tool_call, self.tool_call_end_token)
                if overlap:
                    tool_call = tool_call[:-overlap]
                is_closed = False

            results.append((tool_call, is_closed))

            if end == -1:
                break
        return results

    @staticmethod
    def _extract_tool_id_and_name(
        header: str | None,
    ) -> tuple[str | None, str | None]:
        """Parse ``(tool_id, tool_name)`` from a header
        like ``"functions.get_weather:0"``."""
        if header is None:
            return None, None
        match = re.match(r"(.+:\d+)", header)
        if not match:
            return None, None

        tool_id = match.group(1).strip()
        tool_name = tool_id.split(":")[0].split(".")[-1]
        return tool_id, tool_name

    def _split_tool_call(self, tool_call: str) -> tuple[str | None, str | None]:
        """Split a tool-call body into ``(header, arguments)`` at the argument marker.

        Example::
            'get_weather:0 <|tool_call_argument_begin|>{"c'
            -> ("get_weather:0", '{"c')
        """
        arg_pos = tool_call.find(self.tool_call_arg_token)
        if arg_pos == -1:
            return None, None
        header = tool_call[:arg_pos].strip()
        tool_args = tool_call[arg_pos + len(self.tool_call_arg_token) :]
        return header, tool_args

    def _compute_args_diff(self, index: int, tool_args: str | None) -> str | None:
        """Return new argument text not yet sent for tool `index`, or None."""
        if tool_args is None:
            return None
        prev = self.streamed_args_for_tool[index]
        if len(tool_args) <= len(prev):
            return None
        diff = tool_args[len(prev) :]
        self.streamed_args_for_tool[index] = tool_args
        self.prev_tool_call_arr[index]["arguments"] = tool_args
        return diff

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        try:
            # Extract any content before tool calls.
            content = self._extract_content(current_text)
            tool_calls = self._extract_tool_calls(current_text)
            tool_call_deltas: list[DeltaToolCall] = []

            for i, (tool_call, is_closed) in enumerate(tool_calls):
                # First time seeing tool call at index i.
                if i >= len(self.prev_tool_call_arr):
                    # Initialize streaming state.
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

                header, tool_args = self._split_tool_call(tool_call)

                # Stream back tool name.
                if "name" not in self.prev_tool_call_arr[i]:
                    tool_id, tool_name = self._extract_tool_id_and_name(header)
                    if not tool_name:
                        # Can't skip to tool i+1 if i isn't ready
                        break
                    self.prev_tool_call_arr[i]["name"] = tool_name
                    self.prev_tool_call_arr[i]["id"] = tool_id
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=tool_id,
                            function=DeltaFunctionCall(name=tool_name).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

                # Stream back new tool args by diffing against what was sent.
                args_diff = self._compute_args_diff(i, tool_args)
                if args_diff:
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=args_diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

                # Validate/repair the final args once the call is closed. By
                # now _compute_args_diff has streamed the full raw args, so
                # streamed_args_for_tool[i] == tool_args. If repair only
                # appended (e.g. a missing closing brace), stream the corrective
                # suffix; if it changed already-streamed bytes or fails schema,
                # raise so serving.py returns a 500 and LiteLLM retries.
                if (
                    is_closed
                    and i not in self._validated_tool_idx
                    and tool_args is not None
                ):
                    self._validated_tool_idx.add(i)
                    fn_name = self.prev_tool_call_arr[i].get("name", "unknown")
                    request_id = getattr(request, "request_id", None)
                    (
                        repaired_full,
                        parsed_full,
                        was_repaired,
                        orig_err,
                    ) = _validate_or_repair_args(tool_args, fn_name, request_id)
                    if repaired_full is None:
                        _raise_malformed_tool_call(
                            f"streaming: unrecoverable JSON args for {fn_name} "
                            f"(request_id={request_id}): {orig_err}"
                        )
                    if was_repaired and repaired_full != tool_args:
                        # The streamed body can carry trailing whitespace that
                        # repair normalizes away (the non-streaming regex strips
                        # it via \s* outside the capture group). Compare against
                        # the rstripped prefix so a benign trailing space doesn't
                        # look like a changed prefix. The client's accumulated
                        # args stay valid JSON (json.loads tolerates the space
                        # between the streamed prefix and the corrective suffix).
                        streamed = self.streamed_args_for_tool[i].rstrip()
                        if not repaired_full.startswith(streamed):
                            _raise_malformed_tool_call(
                                f"streaming: repair for {fn_name} changed "
                                f"already-streamed prefix (request_id={request_id})"
                            )
                        issue = self._check_tool_call_issue(
                            repaired_full, fn_name, parsed_full
                        )
                        if issue:
                            _raise_malformed_tool_call(
                                f"streaming: post-repair {issue} for {fn_name} "
                                f"(request_id={request_id}, orig_err={orig_err})"
                            )
                        corrective = repaired_full[len(streamed) :]
                        if corrective:
                            self.streamed_args_for_tool[i] = repaired_full
                            self.prev_tool_call_arr[i]["arguments"] = repaired_full
                            tool_call_deltas.append(
                                DeltaToolCall(
                                    index=i,
                                    function=DeltaFunctionCall(
                                        arguments=corrective
                                    ).model_dump(exclude_none=True),
                                )
                            )

            if content or tool_call_deltas:
                return DeltaMessage(
                    content=content,
                    tool_calls=tool_call_deltas,
                )
            return None

        except Exception as exc:
            # Let MalformedToolCallError propagate so serving.py converts it
            # to a 500 and LiteLLM retries.
            from vllm.entrypoints.openai.chat_completion.serving import (
                MalformedToolCallError,
            )

            if isinstance(exc, MalformedToolCallError):
                raise
            logger.exception("Error trying to handle streaming tool call.")
            return None
