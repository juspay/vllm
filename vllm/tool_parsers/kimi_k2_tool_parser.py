# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# code modified from deepseekv3_tool_parser.py

import json
from collections.abc import Sequence
from typing import Any

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
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)


def _raise_malformed_tool_call(reason: str) -> None:
    """Raise serving.MalformedToolCallError, deferred to break import cycle.

    The parser is imported by serving.py, so we cannot import the exception
    at module load. The exception class lives in serving.py for historical
    reasons (Shivam's earlier degenerate-output detection); raising it here
    bubbles up through serving's existing handlers and becomes a 500 so
    LiteLLM (or any retrying client) retries the request.
    """
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


# Heuristic key extraction from a malformed JSON string. Matches any
# `"key":` pattern at any nesting depth. Used only for telemetry to compare
# what keys were present before vs after json-repair.
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
    """Best-effort structural diff between malformed orig and repaired JSON.

    The orig is malformed so we approximate its keys via regex; the repaired
    side is parsed strictly. Logs which keys appeared/disappeared during
    repair so operators can spot when json-repair drops required fields.
    """
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
) -> tuple[str | None, bool]:
    """Strict JSON parse with json-repair fallback.

    Returns ``(args, was_repaired)`` where:
      * ``(args_str, False)`` - input parses strictly; emitted as-is.
      * ``(repaired, True)`` - input failed but json-repair recovered it.
      * ``(None, True)`` - unrecoverable.

    Callers use ``was_repaired`` to decide whether to apply additional
    safety gates (e.g. schema validation) - json-repair can silently drop
    fields, so the post-repair value should not be trusted blindly.
    """
    try:
        json.loads(args_str)
        return args_str, False
    except json.JSONDecodeError as e:
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
        return None, True
    try:
        repaired = _repair_json(args_str)
        if not repaired:
            return None, True
        json.loads(repaired)
    except Exception as err:
        logger.error("json_repair could not recover args for %s: %s", fn_name, err)
        return None, True
    logger.warning(
        "Repaired Kimi K2 tool args for %s (request_id=%s) len %d -> %d. %s",
        fn_name,
        request_id,
        len(args_str),
        len(repaired),
        _structural_diff(args_str, repaired),
    )
    return repaired, True


class KimiK2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

        # Section-level state management to prevent token leakage
        self.in_tool_section: bool = False
        self.token_buffer: str = ""
        # Buffer size: empirical worst-case for longest marker (~30 chars) * 2
        # + safety margin for unicode + partial overlap. Prevents unbounded growth.
        self.buffer_max_size: int = 49152
        self.section_char_count: int = 0  # Track characters processed in tool section
        self.max_section_chars: int = 49152  # Force exit if section exceeds this
        self._buffer_overflow_logged: bool = False  # Log overflow once per session

        # Support both singular and plural variants
        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"
        self.tool_calls_start_token_variants: list[str] = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_section_begin|>",  # singular variant
        ]
        self.tool_calls_end_token_variants: list[str] = [
            "<|tool_calls_section_end|>",
            "<|tool_call_section_end|>",  # singular variant
        ]

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"

        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<tool_call_id>.+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)"
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<tool_call_id>.+:\d+)\s*")

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        # Get token IDs for all variants
        self.tool_calls_start_token_ids: list[int] = [
            tid
            for variant in self.tool_calls_start_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]
        self.tool_calls_end_token_ids: list[int] = [
            tid
            for variant in self.tool_calls_end_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError(
                "Kimi-K2 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

    def _check_and_strip_markers(self, text: str) -> tuple[str, bool, bool]:
        """
        Check for section begin/end markers in text and strip them.
        Returns: (cleaned_text, found_section_begin, found_section_end)
        """
        found_begin = False
        found_end = False
        cleaned = text

        # Check for section begin markers (any variant)
        for variant in self.tool_calls_start_token_variants:
            if variant in cleaned:
                cleaned = cleaned.replace(variant, "")
                found_begin = True

        # Check for section end markers (any variant)
        for variant in self.tool_calls_end_token_variants:
            if variant in cleaned:
                cleaned = cleaned.replace(variant, "")
                found_end = True
        return cleaned, found_begin, found_end

    def _reset_section_state(self) -> None:
        """Reset state when exiting tool section."""
        self.in_tool_section = False
        self.token_buffer = ""
        self.section_char_count = 0

    def reset_streaming_state(self) -> None:
        """
        Reset all streaming state. Call this between requests to prevent
        state leakage when parser instance is reused.
        """
        # Reset section state
        self._reset_section_state()

        # Reset parent class state
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []

        logger.debug("Streaming state reset")

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

    def _log_malformed_tool_call(
        self,
        *,
        request: ChatCompletionRequest | None,
        function_id: str | None,
        function_name: str | None,
        function_args: str | None,
        model_output: str,
        reason: str,
    ) -> None:
        logger.error(
            "Malformed Kimi K2 tool call generated: request_id=%s, "
            "tool_call_id=%r, tool_name=%r, reason=%s, arguments=%r, "
            "raw_model_output=%r",
            getattr(request, "request_id", None),
            function_id,
            function_name,
            reason,
            function_args,
            model_output,
        )

    def _check_tool_call_issue(
        self, function_args: str, function_name: str
    ) -> str | None:
        """Return the malformation reason for a tool call, or None if OK.

        Skips schema/name checks when the parser was constructed without a
        tools list, since we have nothing to validate against.
        """
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

    def _log_if_malformed_tool_call(
        self,
        *,
        request: ChatCompletionRequest | None,
        function_id: str,
        function_name: str,
        function_args: str,
        model_output: str,
    ) -> str | None:
        """Log if the tool call is malformed; return the reason (or None).

        Returns the reason string so callers can decide whether to drop the
        call. Existing callers that just want observability can ignore the
        return value.
        """
        issue = self._check_tool_call_issue(function_args, function_name)
        if issue:
            self._log_malformed_tool_call(
                request=request,
                function_id=function_id,
                function_name=function_name,
                function_args=function_args,
                model_output=model_output,
                reason=issue,
            )
        return issue

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
                if not function_call_tuples:
                    self._log_malformed_tool_call(
                        request=request,
                        function_id=None,
                        function_name=None,
                        function_args=None,
                        model_output=model_output,
                        reason="tool call section present but no complete "
                        "tool call matched Kimi parser format",
                    )

                request_id = getattr(request, "request_id", None)
                tool_calls = []
                for match in function_call_tuples:
                    function_id, function_args = match
                    # function_id: functions.get_weather:0 or get_weather:0
                    function_name = function_id.split(":")[0].split(".")[-1]
                    repaired_args, was_repaired = _validate_or_repair_args(
                        function_args, function_name, request_id
                    )
                    if repaired_args is None:
                        # Unrecoverable malformed JSON. Log full context for
                        # production telemetry, then raise so serving.py
                        # converts to 500 and LiteLLM retries.
                        self._log_malformed_tool_call(
                            request=request,
                            function_id=function_id,
                            function_name=function_name,
                            function_args=function_args,
                            model_output=model_output,
                            reason="unrecoverable JSON args (json-repair failed)",
                        )
                        _raise_malformed_tool_call(
                            f"unrecoverable JSON args for {function_name} "
                            f"(tool_call_id={function_id}, request_id={request_id})"
                        )
                    # Schema-validate the repaired args. json-repair can
                    # silently drop fields, so we run schema checks on the
                    # post-repair value, not the raw model output.
                    issue = self._log_if_malformed_tool_call(
                        request=request,
                        function_id=function_id,
                        function_name=function_name,
                        function_args=repaired_args,
                        model_output=model_output,
                    )
                    # If json-repair produced JSON that fails schema (e.g.
                    # silently dropped a required field), raise so LiteLLM
                    # retries instead of executing wrong-but-parseable args.
                    # Originally-valid args with schema mismatches still get
                    # emitted (existing behavior) - the gate is specifically
                    # for the new risk introduced by repair.
                    if was_repaired and issue:
                        _raise_malformed_tool_call(
                            f"post-repair {issue} for {function_name} "
                            f"(tool_call_id={function_id}, "
                            f"request_id={request_id})"
                        )
                    tool_calls.append(
                        ToolCall(
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
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)

        # Flag to defer section exit until after tool parsing completes
        deferred_section_exit = False

        # Add delta to buffer for split marker detection
        self.token_buffer += delta_text

        # Enforce buffer size limit to prevent memory issues
        if len(self.token_buffer) > self.buffer_max_size:
            if not self._buffer_overflow_logged:
                logger.warning(
                    "Token buffer exceeded max size (%d bytes), flushing excess. "
                    "This may indicate very long markers or unusual tokenization.",
                    self.buffer_max_size,
                )
                self._buffer_overflow_logged = True
            # Keep only the most recent content that might contain partial markers
            self.token_buffer = self.token_buffer[-self.buffer_max_size // 2 :]

        # Check buffer for section markers (handles split tokens)
        buffered_text, found_section_begin, found_section_end = (
            self._check_and_strip_markers(self.token_buffer)
        )

        # Track section state transitions
        if found_section_begin and not self.in_tool_section:
            logger.debug("Entering tool section")
            self.in_tool_section = True
            self.token_buffer = buffered_text  # Use cleaned buffer
            self.section_char_count = 0  # Reset counter for new section

        if found_section_end and self.in_tool_section:
            logger.debug("Detected section end marker")
            # CRITICAL: Don't exit early if tool_call_end is in this chunk.
            # Tool parser must emit final arguments/close first to avoid dropping
            # the final tool update and leaking tokens into reasoning channel.
            has_tool_end = self.tool_call_end_token_id in delta_token_ids
            if has_tool_end:
                # Defer exit until after tool parsing completes
                deferred_section_exit = True
                logger.debug("Deferring section exit: tool_call_end in same chunk")
                self.token_buffer = buffered_text
            else:
                # No tool call ending, safe to exit immediately
                logger.debug("Exiting tool section")
                self._reset_section_state()
                # Extract any content AFTER the section end marker in delta_text
                # (don't use buffered_text as it contains tool call data)
                post_section_content = ""
                for variant in self.tool_calls_end_token_variants:
                    if variant in delta_text:
                        parts = delta_text.split(variant, 1)
                        if len(parts) > 1:
                            post_section_content = parts[1]
                        break
                if post_section_content.strip():
                    return DeltaMessage(content=post_section_content)
                return DeltaMessage(content="")
        else:
            self.token_buffer = buffered_text

        # Check if any variant of section start token is in current_token_ids
        has_section_token = any(
            tid in current_token_ids for tid in self.tool_calls_start_token_ids
        )

        # Early return: if no section token detected yet, return as reasoning content
        if not has_section_token and not self.in_tool_section:
            logger.debug("No tool call tokens found!")
            # Don't clear buffer - it needs to accumulate partial markers across deltas
            # Buffer overflow is already protected by lines 215-224
            return DeltaMessage(content=delta_text)

        # Strip section markers from delta_text for subsequent processing
        # NOTE: This preprocessing happens BEFORE the regex-based tool call
        # parsing (from PR #24847) to ensure markers are removed cleanly
        # before pattern matching. No double-stripping occurs because
        # section markers and tool call markers are distinct.
        delta_text, _, _ = self._check_and_strip_markers(delta_text)

        # Error recovery: If in tool section for too long, force exit
        if self.in_tool_section:
            self.section_char_count += len(delta_text)
            if self.section_char_count > self.max_section_chars:
                logger.warning(
                    "Tool section exceeded max length (%d chars), forcing exit. "
                    "This may indicate malformed model output.",
                    self.max_section_chars,
                )
                self._reset_section_state()
                # Deferred exit already handled by forced exit above
                # Return remaining content as reasoning (or empty delta if no content)
                return DeltaMessage(content=delta_text if delta_text.strip() else "")

        try:
            # figure out where we are in the parsing by counting tool call
            # start & end tags
            prev_tool_start_count = previous_token_ids.count(
                self.tool_call_start_token_id
            )
            prev_tool_end_count = previous_token_ids.count(self.tool_call_end_token_id)
            cur_tool_start_count = current_token_ids.count(
                self.tool_call_start_token_id
            )
            cur_tool_end_count = current_token_ids.count(self.tool_call_end_token_id)
            tool_call_portion = None
            text_portion = None

            # case: if we're generating text, OR rounding out a tool call
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                # Suppress content between section begin and first tool begin
                # (header noise). Don't suppress content between tools to avoid
                # breaking potential delimiter characters.
                if self.in_tool_section and cur_tool_start_count == 0:
                    logger.debug(
                        "In tool section before first tool, suppressing: %s",
                        delta_text,
                    )
                    # Return empty delta to maintain iterator contract
                    return DeltaMessage(content="")
                logger.debug("Generating text content! skipping tool parsing.")
                return DeltaMessage(content=delta_text)

            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                full_text = current_text + delta_text
                tool_call_portion = (
                    full_text.split(self.tool_call_start_token)[-1]
                    .split(self.tool_call_end_token)[0]
                    .rstrip()
                )
                delta_text = delta_text.split(self.tool_call_end_token)[0].rstrip()
                text_portion = delta_text.split(self.tool_call_end_token)[-1].lstrip()

            # case -- we're starting a new tool call
            if (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count > prev_tool_start_count
            ):
                if len(delta_token_ids) > 1:
                    tool_call_portion = current_text.split(self.tool_call_start_token)[
                        -1
                    ]
                else:
                    tool_call_portion = None
                    delta = None

                text_portion = None

                # set cursors and state appropriately
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            # case -- we're updating an existing tool call
            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count == prev_tool_start_count
            ):
                # get the portion of the text that's the tool call
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                text_portion = None

            # case -- the current tool call is being closed.
            elif (
                cur_tool_start_count == cur_tool_end_count
                and cur_tool_end_count >= prev_tool_end_count
            ):
                if self.prev_tool_call_arr is None or len(self.prev_tool_call_arr) == 0:
                    logger.debug("attempting to close tool call, but no tool call")
                    # Handle deferred section exit before returning
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if diff:
                    diff = (
                        diff.encode("utf-8").decode("unicode_escape")
                        if diff is str
                        else diff
                    )
                    if '"}' not in delta_text:
                        # Handle deferred section exit before returning
                        if deferred_section_exit and self.in_tool_section:
                            self._reset_section_state()
                        return None
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    logger.debug(
                        "Finishing tool and found diff that had not "
                        "been streamed yet: %s",
                        diff,
                    )
                    # Validate the cumulative args at finalization. The
                    # contract matches the non-streaming path: if the args
                    # cannot be safely emitted, raise so serving.py turns
                    # it into a 500 and LiteLLM retries. Cases:
                    #  * strict-parse OK              -> emit as before
                    #  * repair OK + schema OK        -> emit corrected diff
                    #  * repair OK + schema FAIL      -> raise (retry)
                    #  * repair OK + prefix changed   -> raise (retry)
                    #  * repair FAILED                -> raise (retry)
                    already_streamed = self.streamed_args_for_tool[self.current_tool_id]
                    proposed_full = already_streamed + diff
                    fn_name = self.prev_tool_call_arr[self.current_tool_id].get(
                        "name", "unknown"
                    )
                    request_id = getattr(request, "request_id", None)
                    repaired_full, was_repaired = _validate_or_repair_args(
                        proposed_full, fn_name, request_id
                    )
                    if repaired_full is None:
                        logger.error(
                            "Streaming repair failed for %s (request_id=%s); "
                            "raising so client retries. full_len=%d",
                            fn_name,
                            request_id,
                            len(proposed_full),
                        )
                        _raise_malformed_tool_call(
                            f"streaming: unrecoverable JSON args for "
                            f"{fn_name} (request_id={request_id}, "
                            f"full_len={len(proposed_full)})"
                        )
                    if was_repaired and repaired_full != proposed_full:
                        if not repaired_full.startswith(already_streamed):
                            logger.error(
                                "Streaming repair for %s (request_id=%s) "
                                "changed already-streamed prefix; raising "
                                "so client retries. streamed_len=%d "
                                "repaired_len=%d",
                                fn_name,
                                request_id,
                                len(already_streamed),
                                len(repaired_full),
                            )
                            _raise_malformed_tool_call(
                                f"streaming: repair for {fn_name} changed "
                                f"already-streamed prefix "
                                f"(request_id={request_id}, "
                                f"streamed_len={len(already_streamed)}, "
                                f"repaired_len={len(repaired_full)})"
                            )
                        # Prefix-preserving repair. Schema-gate before we
                        # apply: refuse to ship repair-corrupted args.
                        issue = self._check_tool_call_issue(repaired_full, fn_name)
                        if issue:
                            logger.error(
                                "Streaming: post-repair %s for %s "
                                "(request_id=%s); raising so client retries.",
                                issue,
                                fn_name,
                                request_id,
                            )
                            _raise_malformed_tool_call(
                                f"streaming: post-repair {issue} for "
                                f"{fn_name} (request_id={request_id})"
                            )
                        new_diff = repaired_full[len(already_streamed) :]
                        logger.warning(
                            "Streaming repair adjusted final diff for %s "
                            "(request_id=%s): %d -> %d chars",
                            fn_name,
                            request_id,
                            len(diff),
                            len(new_diff),
                        )
                        diff = new_diff
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    # Handle deferred section exit before returning
                    if deferred_section_exit and self.in_tool_section:
                        logger.debug("Completing deferred section exit")
                        self._reset_section_state()
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=diff).model_dump(
                                    exclude_none=True
                                ),
                            )
                        ]
                    )

            # case -- otherwise we're just generating text
            else:
                # Check if we're in tool section - if so, suppress
                if self.in_tool_section:
                    logger.debug("In tool section, suppressing text generation")
                    # Handle deferred section exit before returning
                    if deferred_section_exit:
                        self._reset_section_state()
                    return DeltaMessage(content="")
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                delta = DeltaMessage(tool_calls=[], content=text)
                # Handle deferred section exit before returning
                if deferred_section_exit and self.in_tool_section:
                    self._reset_section_state()
                return delta

            current_tool_call = dict()
            if tool_call_portion:
                current_tool_call_matches = self.stream_tool_call_portion_regex.match(
                    tool_call_portion
                )
                if current_tool_call_matches:
                    tool_id, tool_args = current_tool_call_matches.groups()
                    tool_name = tool_id.split(":")[0].split(".")[-1]
                    current_tool_call["id"] = tool_id.strip()
                    current_tool_call["name"] = tool_name
                    current_tool_call["arguments"] = tool_args
                else:
                    current_tool_call_name_matches = (
                        self.stream_tool_call_name_regex.match(tool_call_portion)
                    )
                    if current_tool_call_name_matches:
                        (tool_id_str,) = current_tool_call_name_matches.groups()
                        tool_name = tool_id_str.split(":")[0].split(".")[-1]
                        current_tool_call["id"] = tool_id_str.strip()
                        current_tool_call["name"] = tool_name
                        current_tool_call["arguments"] = ""
                    else:
                        logger.debug("Not enough token")
                        return None

            # case - we haven't sent the tool name yet. If it's available, send
            #   it. otherwise, wait until it's available.
            if not self.current_tool_name_sent:
                if current_tool_call is None:
                    return None
                function_name: str | None = current_tool_call.get("name")
                tool_id = current_tool_call.get("id")
                if function_name:
                    self.current_tool_name_sent = True
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=tool_id,
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                else:
                    return None

            # case -- otherwise, send the tool call delta

            # if the tool call portion is None, send the delta as text
            if tool_call_portion is None:
                # if there's text but not tool calls, send that -
                # otherwise None to skip chunk
                # CRITICAL: Never return content if we're in a tool section
                if self.in_tool_section:
                    return None
                delta = (
                    DeltaMessage(content=delta_text)
                    if text_portion is not None
                    else None
                )
                return delta

            # now, the nitty-gritty of tool calls
            # now we have the portion to parse as tool call.

            logger.debug(
                "Trying to parse current tool call with ID %s", self.current_tool_id
            )

            # if we're starting a new tool call, push an empty object in as
            #   a placeholder for the arguments
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            # main logic for tool parsing here - compare prev. partially-parsed
            #   JSON to the current partially-parsed JSON
            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )
            cur_arguments = current_tool_call.get("arguments")

            logger.debug("diffing old arguments: %s", prev_arguments)
            logger.debug("against new ones: %s", cur_arguments)

            # case -- no arguments have been created yet. skip sending a delta.
            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
                delta = None

            # case -- prev arguments are defined, but non are now.
            #   probably impossible, but not a fatal error - just keep going
            elif not cur_arguments and prev_arguments:
                logger.error(
                    "should be impossible to have arguments reset "
                    "mid-call. skipping streaming anything."
                )
                delta = None

            # case -- we now have the first info about arguments available from
            #   autocompleting the JSON
            elif cur_arguments and not prev_arguments:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=cur_arguments
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] = cur_arguments

            # last case -- we have an update to existing arguments.
            elif cur_arguments and prev_arguments:
                if (
                    isinstance(delta_text, str)
                    and cur_arguments != prev_arguments
                    and len(cur_arguments) > len(prev_arguments)
                    and cur_arguments.startswith(prev_arguments)
                ):
                    delta_arguments = cur_arguments[len(prev_arguments) :]
                    logger.debug("got diff %s", delta_text)

                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=delta_arguments
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = cur_arguments
                else:
                    delta = None

            # handle saving the state for the current tool into
            # the "prev" list for use in diffing for the next iteration
            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            # Handle deferred section exit after tool parsing completes
            if deferred_section_exit and self.in_tool_section:
                logger.debug("Completing deferred section exit")
                self._reset_section_state()

            return delta

        except Exception as exc:
            # Let MalformedToolCallError propagate so serving.py converts it
            # to a 500 and LiteLLM retries the request.
            from vllm.entrypoints.openai.chat_completion.serving import (
                MalformedToolCallError,
            )

            if isinstance(exc, MalformedToolCallError):
                raise
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.
