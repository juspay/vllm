# Kimi K2.6 Custom Fixes → Nightly Port — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Juspay Kimi-K2.6 custom fixes (reasoning→content fallback, reasoning-parser robustness, tool-arg JSON repair + schema-gate + fail-closed raises, leak/hallucination detection) onto vLLM nightly `0.22.1rc1.dev373+g3d300aecb`, **without** re-introducing the random-tool-call-ID change (keep nightly's history-indexed IDs) and **without** the debug-logging commits.

**Architecture:** Lean into nightly's upstream extension points instead of re-applying the fork's serving.py surgery. Nightly already has (a) the `DelegatingParser.finalize_generation` → `reasoning_parser.get_streaming_fallback_content()` hook for the streaming reasoning→content promotion, and (b) the `NemotronV3ReasoningParser` precedent for the non-streaming swap. So the reasoning→content fix lives in `KimiK2ReasoningParser`. Tool-arg repair/schema/raise lives in `KimiK2ToolParser`, preserving nightly's `id=function_id` scheme. serving.py only gains `MalformedToolCallError` + leak/hallucination detection + clean-500 handlers (the Bundle-B conditionals).

**Tech Stack:** Python 3.12, vLLM nightly, `json-repair>=0.30`, `regex`, pytest.

## Global Constraints

- Base branch: `kimi-2.6-nightly-port` off nightly `3d300aecb`. Do NOT rebase onto the old `v0.19.1` fork.
- **Keep nightly's tool-call ID scheme** (`functions.{name}:{history_idx}` via `make_tool_call_id(id_type="kimi_k2", ...)`). Do NOT port random IDs (`make_tool_call_id()` no-arg). This is what makes the **official** chat template correct on nightly.
- Preserve nightly's stop-sequence buffering guards in the reasoning parser (`if self._end_token not in delta_text: return None`) and the `reasoning_start_str`/`reasoning_end_str` properties.
- Fail-closed (Bundle B): unrecoverable tool-arg JSON, post-repair schema mismatch, leaked `<|...|>` tokens, and repeated hallucinated tool-call patterns → raise `MalformedToolCallError` → clean HTTP 500 so LiteLLM retries.
- Build/test happens on the B300 box (`uv` + `.venv`); this dev box can only `py_compile`.
- New dep floor: `json-repair >= 0.30`.

---

## File Map

- Modify: `requirements/common.txt` — add `json-repair`.
- Modify: `vllm/reasoning/kimi_k2_reasoning_parser.py` — alt-end `</thinking>`, tool-token strip, `get_streaming_fallback_content`.
- Modify: `vllm/tool_parsers/kimi_k2_tool_parser.py` — repair + schema-gate + raise (non-streaming + streaming-at-close), keep nightly IDs.
- Modify: `vllm/entrypoints/openai/chat_completion/serving.py` — `MalformedToolCallError` class, detection helpers, non-streaming fallback + detection, streaming detection, clean-500 handlers in both generators.
- Modify: `tests/reasoning/test_kimi_k2_reasoning_parser.py` — alt-end + fallback cases.
- Modify: `tests/tool_parsers/test_kimi_k2_tool_parser.py` — repair/raise cases (nightly id scheme).

---

### Task 1: Dependency + `MalformedToolCallError` scaffolding

**Files:**
- Modify: `requirements/common.txt`
- Modify: `vllm/entrypoints/openai/chat_completion/serving.py` (module top)

- [ ] **Step 1:** Add to `requirements/common.txt` after the `partial-json-parser` line:
```
json-repair >= 0.30 # used to repair malformed JSON in tool-call args (Kimi K2)
```

- [ ] **Step 2:** In `serving.py`, add `import re` if missing, and a module-level exception class + detection scaffolding near the other module-level defs (after imports):
```python
class MalformedToolCallError(Exception):
    """Raised when Kimi K2 emits unrecoverable tool-call output:
    unrecoverable/malformed tool-call JSON, post-repair schema mismatch,
    leaked control tokens, or repeated hallucinated tool-call-like text.
    serving.py converts this to a clean HTTP 500 so LiteLLM retries."""


# Repeated hallucinated tool-call markers in reasoning => degenerate loop.
_HALLUCINATED_TOOL_CALL_THRESHOLD = 3
_HALLUCINATED_TOOL_CALL_MARKER = "<function_calls>"
# Anthropic-style tool call the model sometimes hallucinates instead of Kimi's
# native <|tool_call_begin|> format. Any occurrence is degenerate.
_HALLUCINATED_ANTHROPIC_TOOL_PATTERN = re.compile(
    r"<function_calls>|<invoke\b|<invoke\b"
)
# Any control token <|...|> leaking into reasoning/content is degenerate.
_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|\S+?\|>")
# A reasoning-close marker leaking into *content* as literal text.
_LEAKED_REASONING_MARKER = "</thinking>"


def _detect_hallucinated_tool_calls_in_reasoning(
    reasoning: str, threshold: int = _HALLUCINATED_TOOL_CALL_THRESHOLD
) -> bool:
    return reasoning.count(_HALLUCINATED_TOOL_CALL_MARKER) >= threshold


def _detect_anthropic_style_tool_calls(text: str) -> bool:
    return _HALLUCINATED_ANTHROPIC_TOOL_PATTERN.search(text) is not None


def _detect_special_tokens_in_text(text: str) -> str | None:
    match = _SPECIAL_TOKEN_PATTERN.search(text)
    return match.group(0) if match else None
```

- [ ] **Step 3:** `python3 -m py_compile vllm/entrypoints/openai/chat_completion/serving.py`. Commit.

---

### Task 2: Reasoning parser — alt-end `</thinking>`, tool-token strip, fallback hook

**Files:**
- Modify: `vllm/reasoning/kimi_k2_reasoning_parser.py`
- Test: `tests/reasoning/test_kimi_k2_reasoning_parser.py`

**Interfaces produced:** `KimiK2ReasoningParser.get_streaming_fallback_content(text, request) -> str | None` (consumed by `DelegatingParser.finalize_generation`).

- [ ] **Step 1 (tests first):** Add to `tests/reasoning/test_kimi_k2_reasoning_parser.py`:
  - `extract_reasoning("<think>reasoning</thinking>answer")` → `("reasoning", "answer")`.
  - `extract_reasoning("<think>just reasoning, no close")` → `("just reasoning, no close", None)`.
  - `is_reasoning_end` for ids containing the `</thinking>` token id → `True`.
  - tool tokens stripped from reasoning: `extract_reasoning("<think>foo<|tool_call_begin|>")[0]` has no `<|tool_call_begin|>`.
  - `get_streaming_fallback_content("<think>unclosed reasoning", req)` → `"unclosed reasoning"`.

- [ ] **Step 2:** Add module regex + `_strip_tool_tokens` near top:
```python
import re
_TOOL_SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|tool_calls?_section_(?:begin|end)\|>"
    r"|<\|tool_call_(?:begin|end)\|>"
    r"|<\|tool_call_argument_begin\|>"
)
```
In the class:
```python
@staticmethod
def _strip_tool_tokens(text: str | None) -> str | None:
    if not text:
        return text
    return _TOOL_SPECIAL_TOKEN_PATTERN.sub("", text)
```

- [ ] **Step 3:** In `__init__`, after `self._tool_section_start_token` block, add the alt-end token:
```python
self._alt_end_token = "</thinking>"
self._alt_end_token_id = self.vocab.get(self._alt_end_token)
```
(`None` is fine if the tokenizer lacks it — all uses are `is not None`-guarded.)

- [ ] **Step 4:** Extend `is_reasoning_end` backward scan to also treat `_alt_end_token_id` as an end (insert alongside the `end_token_id` check):
```python
if self._alt_end_token_id is not None and input_ids[i] == self._alt_end_token_id:
    return True
```

- [ ] **Step 5:** Extend `is_reasoning_end_streaming` to also return True when `_alt_end_token_id in delta_ids_set`.

- [ ] **Step 6:** Extend `extract_content_ids` with an alt-end branch mirroring the `_end_token_id` branch (search `_alt_end_token_id`, return ids after it).

- [ ] **Step 7:** In `extract_reasoning`, add an alt-end branch after the `</think>` branch (before the tool-section branch), and strip tool tokens from the returned reasoning:
```python
alt_end_index = model_output.find(self._alt_end_token)
if alt_end_index != -1:
    return (
        self._strip_tool_tokens(model_output[start_token_index:alt_end_index]),
        model_output[alt_end_index + len(self._alt_end_token):] or None,
    )
```
Also wrap the existing `</think>`, tool-section, and "still reasoning" return values' reasoning component in `self._strip_tool_tokens(...)`.

- [ ] **Step 8:** In `extract_reasoning_streaming`, KEEP nightly's structure + buffering guards. Add `_alt_end_token_id` to the single-special-token skip list, and add an alt-end delta block mirroring the `_end_token_id` block (with the same `if self._alt_end_token not in delta_text: return None` guard). Wrap the `reasoning=` field in `_strip_tool_tokens(...)` in the final `return DeltaMessage(reasoning=...)`.

- [ ] **Step 9:** Add the fallback hook (NemotronV3 pattern). Only fires via `finalize_generation` when `state.reasoning_ended` is False (i.e. model never closed think):
```python
def get_streaming_fallback_content(
    self, text: str, request: "ChatCompletionRequest | ResponsesRequest"
) -> str | None:
    """Promote accumulated reasoning into content on the terminal streaming
    delta when reasoning never ended (model produced no </think> and no tool
    section), so OpenAI clients don't receive null content."""
    if self._identity_parser is not None:
        return None
    reasoning, content = self.extract_reasoning(text, request)
    if content is None and reasoning:
        return reasoning
    return None
```

- [ ] **Step 10:** `py_compile`; run reasoning tests on box: `.venv/bin/python -m pytest tests/reasoning/test_kimi_k2_reasoning_parser.py -v`. Commit.

---

### Task 3: Tool parser — JSON repair + schema-gate + fail-closed raise (keep nightly IDs)

**Files:**
- Modify: `vllm/tool_parsers/kimi_k2_tool_parser.py`
- Test: `tests/tool_parsers/test_kimi_k2_tool_parser.py`

**Interfaces consumed:** `serving.MalformedToolCallError` (lazy import to break cycle).

- [ ] **Step 1 (tests first):** Add to `tests/tool_parsers/test_kimi_k2_tool_parser.py` (note: IDs are `functions.X:0`, NOT random):
  - valid args pass through unchanged, `id == "functions.get_weather:0"`.
  - repairable args (`{"a": 1,}` trailing comma) → emitted repaired, no raise.
  - unrecoverable args (`{"a": }`) → raises `MalformedToolCallError`.
  - post-repair schema mismatch (repaired drops a `required` field) with `tools=[schema]` → raises.
  - unknown tool name with **valid** JSON and `tools` set → still emitted (NOT gated; only post-repair path checks names).

- [ ] **Step 2:** Add imports + helpers (copy verbatim from fork `vllm/tool_parsers/kimi_k2_tool_parser.py`): `import json`, `from typing import Any, NoReturn`, the `_raise_malformed_tool_call` lazy-import helper, the `json_repair` optional import block, `_JSON_KEY_PATTERN`, `_all_keys_in_value`, `_structural_diff`, `_validate_or_repair_args`. These are architecture-independent and port as-is.

- [ ] **Step 3:** Add the schema helpers to `KimiK2ToolParser` (copy verbatim from fork): `_get_tool_schema`, `_json_type_name`, `_matches_json_type`, `_validate_schema_subset`, `_check_tool_call_issue`, and `_extract_function_name`.

- [ ] **Step 4 (non-streaming):** In `extract_tool_calls`, inside the `for match in function_call_tuples` loop, after computing `function_name = function_id.split(":")[0].split(".")[-1]`, insert repair+validate and emit the repaired args. **Keep `id=function_id`** (nightly scheme):
```python
request_id = getattr(request, "request_id", None)
repaired_args, parsed_args, was_repaired, orig_err = _validate_or_repair_args(
    function_args, function_name, request_id
)
if repaired_args is None:
    _raise_malformed_tool_call(
        f"unrecoverable JSON args for {function_name} "
        f"(tool_call_id={function_id}, request_id={request_id}): {orig_err}"
    )
if was_repaired:
    issue = self._check_tool_call_issue(repaired_args, function_name, parsed_args)
    if issue:
        _raise_malformed_tool_call(
            f"post-repair {issue} for {function_name} "
            f"(tool_call_id={function_id}, request_id={request_id}, orig_err={orig_err})"
        )
tool_calls.append(
    ToolCall(
        id=function_id,  # nightly scheme — do NOT use make_tool_call_id()
        type="function",
        function=FunctionCall(name=function_name, arguments=repaired_args),
    )
)
```
In the `except Exception` handler, re-raise `MalformedToolCallError` (lazy import + `isinstance` check) so it isn't swallowed into "no tools".

- [ ] **Step 5 (streaming, validate-at-close):** Change `_extract_tool_calls` to return `list[tuple[str, bool]]` where the bool is `is_closed = (end != -1)`. Add `self._validated_tool_idx: set[int] = set()` in `__init__`. In `extract_tool_calls_streaming`, iterate `for i, (tool_call, is_closed) in enumerate(tool_calls)`. After the existing name/args-diff logic for tool `i`, when `is_closed and i not in self._validated_tool_idx`:
```python
self._validated_tool_idx.add(i)
header, tool_args = self._split_tool_call(tool_call)
if tool_args is not None:
    fn_name = self.prev_tool_call_arr[i].get("name", "unknown")
    request_id = getattr(request, "request_id", None)
    repaired, parsed, was_repaired, orig_err = _validate_or_repair_args(
        tool_args, fn_name, request_id
    )
    if repaired is None:
        _raise_malformed_tool_call(
            f"streaming: unrecoverable JSON args for {fn_name} "
            f"(request_id={request_id}): {orig_err}"
        )
    if was_repaired and repaired != tool_args:
        streamed = self.streamed_args_for_tool[i]
        if not repaired.startswith(streamed):
            _raise_malformed_tool_call(
                f"streaming: repair for {fn_name} changed already-streamed "
                f"prefix (request_id={request_id})"
            )
        issue = self._check_tool_call_issue(repaired, fn_name, parsed)
        if issue:
            _raise_malformed_tool_call(
                f"streaming: post-repair {issue} for {fn_name} "
                f"(request_id={request_id}, orig_err={orig_err})"
            )
        corrective = repaired[len(streamed):]
        if corrective:
            self.streamed_args_for_tool[i] = repaired
            self.prev_tool_call_arr[i]["arguments"] = repaired
            tool_call_deltas.append(
                DeltaToolCall(
                    index=i,
                    function=DeltaFunctionCall(arguments=corrective).model_dump(
                        exclude_none=True
                    ),
                )
            )
```
Re-raise `MalformedToolCallError` in the `except Exception` handler (lazy import + `isinstance`). Reset `self._validated_tool_idx` wherever `prev_tool_call_arr`/`streamed_args_for_tool` are reset.

> **Deliberate simplification vs fork:** the fork tried prefix-compatible re-streaming in its old state machine; here we re-stream only the appended corrective suffix (the common unclosed-brace case) and raise on any mid-string change. Same fail-closed contract, simpler/safer in nightly's parser.

- [ ] **Step 6:** `py_compile`; run on box: `.venv/bin/python -m pytest tests/tool_parsers/test_kimi_k2_tool_parser.py -v`. Commit.

---

### Task 4: serving.py — non-streaming fallback + leak/hallucination detection + clean 500s

**Files:**
- Modify: `vllm/entrypoints/openai/chat_completion/serving.py`

- [ ] **Step 1 (non-streaming):** In `chat_completion_full_generator`, after `reasoning, content, tool_calls = parser.parse(...)` (~line 953-963), and before `auto_tools_called = False`, add detection + fallback:
```python
if parser is not None and self.reasoning_parser is not None:
    if reasoning:
        if _detect_hallucinated_tool_calls_in_reasoning(reasoning):
            raise MalformedToolCallError(
                "repeated hallucinated tool-call patterns in reasoning"
            )
        if _detect_anthropic_style_tool_calls(reasoning):
            raise MalformedToolCallError(
                "model hallucinated Anthropic-style tool calls in reasoning"
            )
        leaked = _detect_special_tokens_in_text(reasoning)
        if leaked:
            raise MalformedToolCallError(
                f"special token {leaked!r} leaked into reasoning"
            )
    if content:
        leaked = _detect_special_tokens_in_text(content)
        if leaked:
            raise MalformedToolCallError(
                f"special token {leaked!r} leaked into content"
            )
    # reasoning -> content fallback (model never closed </think>)
    if not content and reasoning and not tool_calls:
        content = reasoning
```

- [ ] **Step 2 (non-streaming 500):** Wrap the body of the `for output in final_res.outputs:` loop in `try: ... except MalformedToolCallError as e:` that returns:
```python
return self.create_error_response(
    str(e), err_type="InternalServerError",
    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
)
```

- [ ] **Step 3 (streaming accumulation + detection):** In `chat_completion_stream_generator`, near `previous_texts = [""] * num_choices`, add `accumulated_reasoning_arr = [""] * num_choices`. After `delta_message` is finalized non-None (after line ~662) and only when `self.reasoning_parser is not None`:
```python
if delta_message.reasoning:
    accumulated_reasoning_arr[i] += delta_message.reasoning
    if _detect_hallucinated_tool_calls_in_reasoning(accumulated_reasoning_arr[i]):
        raise MalformedToolCallError("repeated hallucinated tool-call patterns in reasoning")
    if _detect_anthropic_style_tool_calls(delta_message.reasoning):
        raise MalformedToolCallError("model hallucinated Anthropic-style tool calls in reasoning")
    leaked = _detect_special_tokens_in_text(delta_message.reasoning)
    if leaked:
        raise MalformedToolCallError(f"special token {leaked!r} leaked into reasoning")
if delta_message.content:
    leaked = _detect_special_tokens_in_text(delta_message.content)
    if leaked:
        raise MalformedToolCallError(f"special token {leaked!r} leaked into content")
    if _LEAKED_REASONING_MARKER in delta_message.content:
        raise MalformedToolCallError("model leaked </thinking> into content")
```
(The streaming reasoning→content fallback itself needs **no** code here — `KimiK2ReasoningParser.get_streaming_fallback_content` + nightly's `finalize_generation` handle it.)

- [ ] **Step 4 (streaming 500):** Add `except MalformedToolCallError as e:` BEFORE the existing `except Exception as e:` in the stream generator, emitting a clean streaming error then `[DONE]`:
```python
except MalformedToolCallError as e:
    logger.warning("Malformed Kimi tool call (streaming): %s", e)
    data = self.create_streaming_error_response(
        str(e), err_type="InternalServerError",
    )
    yield f"data: {data}\n\n"
```

- [ ] **Step 5:** `py_compile`. On box: run `pytest tests/entrypoints/openai -k kimi` if present, plus a live smoke test (Task 5). Commit.

---

### Task 5: Verification + memory

- [ ] **Step 1:** `python3 -m py_compile` all four modified `.py` files (dev box).
- [ ] **Step 2 (on B300 box):** `uv pip install json-repair>=0.30`; rebuild `VLLM_USE_PRECOMPILED=1 uv pip install -e .`; run `.venv/bin/python -m pytest tests/reasoning/test_kimi_k2_reasoning_parser.py tests/tool_parsers/test_kimi_k2_tool_parser.py -v`.
- [ ] **Step 3 (on B300 box):** Switch serving to the **official** chat template (+ optional Juspay system-prompt block); confirm `model_type ∈ {kimi_k2, kimi_k25}` or pass `--hf-overrides '{"model_type":"kimi_k2"}'`. Live smoke: a prompt that previously hung (long reasoning, no `</think>`) now returns non-null `content`; a multi-tool prompt returns `functions.{name}:{idx}` IDs.
- [ ] **Step 4:** Update memory: fork is NOT fully droppable on nightly (reasoning→content + repair still needed); custom template reverts nightly's ID fix; `_KIMI_MODEL_TYPES` doesn't include k2.6.

---

## Self-Review

- **Coverage:** reasoning→content (Task 2 hook + Task 4 non-stream), alt-end `</thinking>` (Task 2), tool-token strip (Task 2), JSON repair + schema-gate + raise (Task 3), leak/hallucination 5XX (Task 4), `MalformedToolCallError` + 500s (Tasks 1/3/4), dep (Task 1). Dropped by design: random IDs, debug logging.
- **ID scheme:** every `ToolCall` keeps `id=function_id`; no `make_tool_call_id()` no-arg call introduced. ✓
- **Nightly guards preserved:** buffering `return None` guards + `reasoning_start_str`/`reasoning_end_str` retained in Task 2 (additive merge, not replace). ✓
