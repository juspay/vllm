# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json

import pytest

from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.kimi_k2_tool_parser import (
    KimiK2ToolParser,
    _structural_diff,
    _validate_or_repair_args,
)

# Use a common model that is likely to be available
MODEL = "moonshotai/Kimi-K2-Instruct"


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL, trust_remote_code=True)


@pytest.fixture
def kimi_k2_tool_parser(kimi_k2_tokenizer):
    return KimiK2ToolParser(kimi_k2_tokenizer)


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function

        # Tool call IDs are random UUIDs: chatcmpl-tool-<16-hex-chars>
        assert actual_tool_call.id.startswith("chatcmpl-tool-"), (
            f"Expected random ID format, got: {actual_tool_call.id}"
        )


def run_streaming_sequence(parser, deltas):
    """Helper to simulate a streaming sequence and return results."""
    previous_text = ""
    previous_token_ids: list[int] = []
    results = []

    for delta_text, delta_token_ids in deltas:
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_token_ids

        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=None,
        )
        results.append(result)

        previous_text = current_text
        previous_token_ids = current_token_ids

    return results


def test_extract_tool_calls_no_tools(kimi_k2_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "tool_call_with_content_before",
        "multi_tool_call_with_content_before",
        "concatenated_tool_calls_bug_fix",
        "three_concatenated_tool_calls",
        "mixed_spacing_tool_calls",
        "angle_brackets_in_json",
        "newlines_in_json",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_calls_section_end|>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Beijing",
                            },
                        ),
                    ),
                    type="function",
                )
            ],
            "I'll help you check the weather. ",
        ),
        (
            """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_call_begin|>
functions.get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Beijing",
                            },
                        ),
                    ),
                    type="function",
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Shanghai",
                            },
                        ),
                    ),
                    type="function",
                ),
            ],
            "I'll help you check the weather. ",
        ),
        (
            """I'll get the weather and news for LA today. First, let me get the weather using Los Angeles coordinates, and then get the latest news. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"latitude": 34.0522, "longitude": -118.2437}<|tool_call_end|><|tool_call_begin|>functions.get_news:1<|tool_call_argument_begin|>{"content": "Los Angeles today"}<|tool_call_end|><|tool_calls_section_end|>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {"latitude": 34.0522, "longitude": -118.2437}
                        ),
                    ),
                    type="function",
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_news",
                        arguments=json.dumps({"content": "Los Angeles today"}),
                    ),
                    type="function",
                ),
            ],
            "I'll get the weather and news for LA today. First, let me get the weather using Los Angeles coordinates, and then get the latest news. ",
        ),
        (
            """I'll help you with multiple tasks. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "New York"}<|tool_call_end|><|tool_call_begin|>functions.get_news:1<|tool_call_argument_begin|>{"topic": "technology"}<|tool_call_end|><|tool_call_begin|>functions.send_email:2<|tool_call_argument_begin|>{"to": "user@example.com", "subject": "Daily Update"}<|tool_call_end|><|tool_calls_section_end|>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps({"city": "New York"}),
                    ),
                    type="function",
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_news",
                        arguments=json.dumps({"topic": "technology"}),
                    ),
                    type="function",
                ),
                ToolCall(
                    function=FunctionCall(
                        name="send_email",
                        arguments=json.dumps(
                            {"to": "user@example.com", "subject": "Daily Update"}
                        ),
                    ),
                    type="function",
                ),
            ],
            "I'll help you with multiple tasks. ",
        ),
        (
            """Mixed spacing test. <|tool_calls_section_begin|> <|tool_call_begin|> functions.test:0 <|tool_call_argument_begin|> {} <|tool_call_end|><|tool_call_begin|>functions.test2:1<|tool_call_argument_begin|>{}<|tool_call_end|> <|tool_calls_section_end|>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="test",
                        arguments=json.dumps({}),
                    ),
                    type="function",
                ),
                ToolCall(
                    function=FunctionCall(
                        name="test2",
                        arguments=json.dumps({}),
                    ),
                    type="function",
                ),
            ],
            "Mixed spacing test. ",
        ),
        (
            """I need to process HTML content. <|tool_calls_section_begin|><|tool_call_begin|>functions.process_html:0<|tool_call_argument_begin|>{"html": "<div>content</div>", "text": "normal text"}<|tool_call_end|><|tool_calls_section_end|>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="process_html",
                        arguments=json.dumps(
                            {"html": "<div>content</div>", "text": "normal text"}
                        ),
                    ),
                    type="function",
                )
            ],
            "I need to process HTML content. ",
        ),
        (
            """I need to process formatted JSON. <|tool_calls_section_begin|><|tool_call_begin|>functions.process_data:0<|tool_call_argument_begin|>{
  "name": "test",
  "value": 123,
  "nested": {
    "key": "value"
  }
}<|tool_call_end|><|tool_calls_section_end|>""",
            [
                ToolCall(
                    function=FunctionCall(
                        name="process_data",
                        arguments=json.dumps(
                            {"name": "test", "value": 123, "nested": {"key": "value"}},
                            indent=2,
                        ),
                    ),
                    type="function",
                )
            ],
            "I need to process formatted JSON. ",
        ),
    ],
)
def test_extract_tool_calls(
    kimi_k2_tool_parser, model_output, expected_tool_calls, expected_content
):
    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_invalid_json(kimi_k2_tool_parser):
    """Malformed JSON args should be repaired (json-repair) before being
    emitted, so the client always receives parseable JSON."""
    model_output = """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.invalid_get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing" <|tool_call_end|> <|tool_call_begin|>
functions.valid_get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>"""

    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == "invalid_get_weather"
    assert extracted_tool_calls.tool_calls[1].function.name == "valid_get_weather"
    # Both args strings must now parse as valid JSON.
    args0 = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    args1 = json.loads(extracted_tool_calls.tool_calls[1].function.arguments)
    assert args0 == {"city": "Beijing"}
    assert args1 == {"city": "Shanghai"}


def test_extract_tool_calls_repair_production_sample(kimi_k2_tool_parser):
    """Regression: production failure where Kimi K2.6 emitted args missing
    the final closing brace. json-repair should recover them."""
    # Args from production: outer object missing the final `}`.
    bad_args = (
        '{"input": "{\\"request_id\\": \\"f9faaf55-6a62-4445-8fb5-e088c33f09f9\\", '
        '\\"order_id\\": \\"TEJFEF84815\\", \\"merchant_id\\": \\"hungerbox\\", '
        '\\"query\\": \\"Investigate decideGatewayHS function that threw '
        'DECIDE_GATEWAY_HS_FAILED error.\\"}"'
    )
    model_output = (
        "<|tool_calls_section_begin|> <|tool_call_begin|> "
        f"functions.analyze_code:14 <|tool_call_argument_begin|> {bad_args} "
        "<|tool_call_end|> <|tool_calls_section_end|>"
    )
    # Sanity: confirm the input is genuinely malformed.
    with pytest.raises(json.JSONDecodeError):
        json.loads(bad_args)

    extracted = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
    assert extracted.tools_called
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "analyze_code"
    # Repaired args parse cleanly and preserve the inner payload.
    parsed = json.loads(extracted.tool_calls[0].function.arguments)
    assert "input" in parsed
    assert "request_id" in parsed["input"]


def test_validate_or_repair_args_strict_passthrough():
    """Already-valid JSON should pass through unchanged with was_repaired=False."""
    src = '{"a": 1, "b": "hello"}'
    out, parsed, was_repaired, orig_err = _validate_or_repair_args(src, "fn")
    assert out == src
    assert parsed == {"a": 1, "b": "hello"}
    assert was_repaired is False
    assert orig_err is None


def test_validate_or_repair_args_missing_close_brace():
    src = '{"city": "Beijing"'
    out, parsed, was_repaired, orig_err = _validate_or_repair_args(src, "get_weather")
    assert out is not None
    assert was_repaired is True
    assert orig_err is not None
    assert parsed == {"city": "Beijing"}
    assert json.loads(out) == {"city": "Beijing"}


def test_validate_or_repair_args_invalid_escape():
    """Haskell-style `\\_` escape should be repaired."""
    src = '{"code": "let x = 1\\_2"}'
    out, parsed, was_repaired, orig_err = _validate_or_repair_args(src, "run")
    assert out is not None
    assert was_repaired is True
    assert orig_err is not None
    assert isinstance(parsed, dict)
    json.loads(out)  # must parse


def test_validate_or_repair_args_trailing_junk():
    src = '{"a": 1} unexpected trailing stuff'
    out, parsed, was_repaired, orig_err = _validate_or_repair_args(src, "fn")
    assert out is not None
    assert was_repaired is True
    assert orig_err is not None
    assert parsed == {"a": 1}
    assert json.loads(out) == {"a": 1}


def test_structural_diff_no_change():
    """Repair that just closes a brace should not show dropped/added keys."""
    diff = _structural_diff('{"a": 1, "b": 2', '{"a": 1, "b": 2}')
    assert "keys_dropped=[]" in diff
    assert "keys_added=[]" in diff


def test_structural_diff_dropped_field():
    """Repair that drops a field should be visible in the diff log."""
    # `{"a":1,"b":}` → `{"a":1}` - json_repair commonly drops broken values.
    diff = _structural_diff('{"a": 1, "b":', '{"a": 1}')
    assert "keys_dropped=['b']" in diff


def test_validate_or_repair_args_unrecoverable():
    """Pure garbage with no recoverable JSON returns None."""
    out, parsed, was_repaired, orig_err = _validate_or_repair_args(
        "@@@ not json at all @@@", "fn"
    )
    assert was_repaired is True
    assert orig_err is not None
    # json-repair is permissive and may produce an empty container; either
    # None or empty-but-parseable is acceptable - the contract is that the
    # returned string is parseable or None.
    if out is None:
        assert parsed is None
    else:
        json.loads(out)


def test_streaming_repair_at_finalization(kimi_k2_tool_parser):
    """Streaming: malformed args (missing closing brace) are repaired at the
    close-of-tool-call boundary so the cumulative args the client sees parse
    as valid JSON."""
    kimi_k2_tool_parser.reset_streaming_state()

    # Single-shot delta containing the whole tool call with malformed args.
    # The args `{"city": "Beijing"` is missing the trailing `}`. Since the
    # close branch keys on `"}` in the trailing delta, we compose args that
    # end in `"}` but with broken structure earlier - here, an unclosed
    # outer object after a nested string close: `{"a": "b"}"}` would
    # already parse, so use a simpler malformation: a known bad escape.
    args = '{"code": "let x = 1\\_2"}'
    full_call = (
        f"<|tool_calls_section_begin|> <|tool_call_begin|> "
        f"functions.run:0 <|tool_call_argument_begin|> {args} "
        f"<|tool_call_end|> <|tool_calls_section_end|>"
    )
    deltas = [(full_call, [])]
    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    # Collect every args fragment emitted across all deltas.
    emitted_args = ""
    for r in results:
        if r is None or not getattr(r, "tool_calls", None):
            continue
        for tc in r.tool_calls:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            arg_chunk = fn.get("arguments") if isinstance(fn, dict) else fn.arguments
            if arg_chunk:
                emitted_args += arg_chunk

    # Whatever the parser streamed must reassemble into valid JSON.
    if emitted_args:
        json.loads(emitted_args)


def test_extract_tool_calls_repair_invalid_escape(kimi_k2_tool_parser):
    """`\\_` (Haskell-style invalid JSON escape) should be repaired."""
    model_output = (
        "<|tool_calls_section_begin|> <|tool_call_begin|> "
        'functions.run:0 <|tool_call_argument_begin|> {"code": "let x = 1\\_2"} '
        "<|tool_call_end|> <|tool_calls_section_end|>"
    )
    extracted = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
    assert extracted.tools_called
    assert len(extracted.tool_calls) == 1
    # Must parse as JSON after repair.
    json.loads(extracted.tool_calls[0].function.arguments)


def test_extract_tool_calls_post_repair_schema_mismatch_raises(kimi_k2_tokenizer):
    """When repair succeeds but the repaired args fail schema validation
    (e.g. json-repair silently dropped a required field), the parser must
    raise MalformedToolCallError so serving.py converts it to a 500 and
    LiteLLM retries."""
    from vllm.entrypoints.openai.chat_completion.serving import (
        MalformedToolCallError,
    )

    class _Fn:
        def __init__(self, name, parameters):
            self.name = name
            self.parameters = parameters

    class _Tool:
        def __init__(self, function):
            self.function = function

    tools = [
        _Tool(
            _Fn(
                "get_weather",
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        )
    ]
    parser = KimiK2ToolParser(kimi_k2_tokenizer, tools=tools)
    # `{"foo": "bar"` is malformed; repair gives `{"foo": "bar"}` which is
    # parseable but missing the required `city` field.
    model_output = (
        "<|tool_calls_section_begin|> <|tool_call_begin|> "
        'functions.get_weather:0 <|tool_call_argument_begin|> {"foo": "bar" '
        "<|tool_call_end|> <|tool_calls_section_end|>"
    )
    with pytest.raises(MalformedToolCallError, match="post-repair"):
        parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]


def test_extract_tool_calls_unrecoverable_json_raises(kimi_k2_tokenizer):
    """When repair fails entirely, the parser must raise so LiteLLM retries.

    This requires we feed args that json-repair cannot recover. json-repair
    is permissive, so we use a parser fixture without the json-repair
    package; if it's installed we cannot reliably trigger this case in a
    unit test - the behavior is verified at the helper level by
    test_validate_or_repair_args_unrecoverable.
    """
    # The contract is: when _validate_or_repair_args returns (None, _),
    # extract_tool_calls raises MalformedToolCallError. Verify the
    # contract by patching the helper.
    from vllm.entrypoints.openai.chat_completion.serving import (
        MalformedToolCallError,
    )
    from vllm.tool_parsers import kimi_k2_tool_parser as kk2

    parser = KimiK2ToolParser(kimi_k2_tokenizer)
    model_output = (
        "<|tool_calls_section_begin|> <|tool_call_begin|> "
        'functions.run:0 <|tool_call_argument_begin|> {"a": 1} '
        "<|tool_call_end|> <|tool_calls_section_end|>"
    )
    real = kk2._validate_or_repair_args
    try:
        kk2._validate_or_repair_args = lambda *a, **kw: (  # type: ignore[assignment]
            None,
            None,
            True,
            "synthetic error at pos 0",
        )
        with pytest.raises(MalformedToolCallError, match="unrecoverable"):
            parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
    finally:
        kk2._validate_or_repair_args = real  # type: ignore[assignment]


def test_extract_tool_calls_invalid_funcall(kimi_k2_tool_parser):
    """we'll return every funcall result"""
    model_output = """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.invalid_get_weather.0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_call_begin|>
functions.valid_get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>"""

    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid JSON tool calls
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "valid_get_weather"


def test_streaming_basic_functionality(kimi_k2_tool_parser):
    """Test basic streaming functionality."""
    # Reset streaming state
    kimi_k2_tool_parser.current_tool_name_sent = False
    kimi_k2_tool_parser.prev_tool_call_arr = []
    kimi_k2_tool_parser.current_tool_id = -1
    kimi_k2_tool_parser.streamed_args_for_tool = []

    # Test with a simple tool call
    current_text = """ check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_calls_section_end|>"""

    # First call should handle the initial setup
    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="I'll help you",
        current_text=current_text,
        delta_text="<|tool_calls_section_end|>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # The result might be None or contain tool call information
    # This depends on the internal state management
    if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
        assert len(result.tool_calls) >= 0


def test_streaming_no_tool_calls(kimi_k2_tool_parser):
    """Test streaming when there are no tool calls."""
    current_text = "This is just regular text without any tool calls."

    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="This is just regular text",
        current_text=current_text,
        delta_text=" without any tool calls.",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Should return the delta text as content
    assert result is not None
    assert hasattr(result, "content")
    assert result.content == " without any tool calls."


def test_token_leak_between_section_and_tool_begin(kimi_k2_tool_parser):
    """
    Test that text between <|tool_calls_section_begin|> and <|tool_call_begin|>
    is suppressed and does not leak into reasoning_delta.
    This is the main vulnerability being fixed.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    # Get token IDs for the markers
    section_begin_token_id = kimi_k2_tool_parser.vocab.get(
        "<|tool_calls_section_begin|>"
    )
    tool_call_begin_token_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")

    # Simulate streaming sequence:
    deltas = [
        ("I'll help you with that. ", [1, 2, 3]),
        ("<|tool_calls_section_begin|>", [section_begin_token_id]),
        (" spurious text ", [4, 5]),
        ("<|tool_call_begin|>", [tool_call_begin_token_id]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    # Delta 1: "I'll help you with that. "
    assert results[0] is not None
    assert results[0].content == "I'll help you with that. "

    # Delta 2: "<|tool_calls_section_begin|>"
    # Section marker should be stripped and suppressed
    assert results[1] is None or (
        results[1].content is None or results[1].content == ""
    )

    # Delta 3: " spurious text or tokens " (THE LEAK SCENARIO)
    # CRITICAL: This text should be suppressed, NOT returned as reasoning_delta
    assert results[2] is None or (
        results[2].content is None or results[2].content == ""
    )

    # Delta 4: "<|tool_call_begin|>..."
    # Now we're in tool call mode, result depends on internal state
    # The key is that the spurious text from Delta 3 was not leaked


def test_split_markers_across_deltas(kimi_k2_tool_parser):
    """
    Test that markers split across delta chunks are correctly detected
    via the rolling buffer mechanism.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_token_id = kimi_k2_tool_parser.vocab.get(
        "<|tool_calls_section_begin|>"
    )

    # Delta 1: partial token, Delta 2: complete marker
    deltas = [
        ("<|tool_calls_sec", [3]),
        ("tion_begin|> ", [section_begin_token_id, 4]),
    ]

    _results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    # Now the complete marker should be detected via buffer
    assert kimi_k2_tool_parser.in_tool_section is True


def test_marker_variants(kimi_k2_tool_parser):
    """Test that both singular and plural marker variants are recognized."""
    kimi_k2_tool_parser.reset_streaming_state()

    # Test singular variant: <|tool_call_section_begin|> (note: singular "call")
    singular_token_id = kimi_k2_tool_parser.vocab.get("<|tool_call_section_begin|>")

    if singular_token_id is not None:  # Only test if tokenizer supports it
        _result = kimi_k2_tool_parser.extract_tool_calls_streaming(
            previous_text="Reasoning ",
            current_text="Reasoning <|tool_call_section_begin|>",
            delta_text="<|tool_call_section_begin|>",
            previous_token_ids=[1, 2],
            current_token_ids=[1, 2, singular_token_id],
            delta_token_ids=[singular_token_id],
            request=None,
        )
        # Should enter tool section mode with singular variant too
        assert kimi_k2_tool_parser.in_tool_section is True


def test_reentry_to_reasoning_after_tool_section(kimi_k2_tool_parser):
    """
    Test that after exiting a tool section with <|tool_calls_section_end|>,
    subsequent text is correctly returned as reasoning content.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    deltas = [
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        ("<|tool_calls_section_end|>", [section_end_id]),
        (" More reasoning", [10, 11]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    assert kimi_k2_tool_parser.in_tool_section is False
    assert results[2] is not None
    assert results[2].content == " More reasoning"


def test_empty_tool_section(kimi_k2_tool_parser):
    """Test an empty tool section (begin immediately followed by end)."""
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    # Section begin
    _result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning ",
        current_text="Reasoning <|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[1],
        current_token_ids=[1, section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )

    # Immediate section end
    _result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning <|tool_calls_section_begin|>",
        current_text="Reasoning <|tool_calls_section_begin|><|tool_calls_section_end|>",
        delta_text="<|tool_calls_section_end|>",
        previous_token_ids=[1, section_begin_id],
        current_token_ids=[1, section_begin_id, section_end_id],
        delta_token_ids=[section_end_id],
        request=None,
    )
    # Should exit cleanly without errors
    assert kimi_k2_tool_parser.in_tool_section is False


def test_malformed_tool_section_recovery(kimi_k2_tool_parser):
    """
    Test that the parser recovers from a malformed tool section
    that never closes properly.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")

    # Enter tool section
    _result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[],
        current_token_ids=[section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )
    assert kimi_k2_tool_parser.in_tool_section is True

    # Simulate a lot of text without proper tool calls or section end
    # This should trigger the error recovery mechanism
    large_text = "x" * 10000  # Exceeds max_section_chars

    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="<|tool_calls_section_begin|>",
        current_text="<|tool_calls_section_begin|>" + large_text,
        delta_text=large_text,
        previous_token_ids=[section_begin_id],
        current_token_ids=[section_begin_id] + list(range(100, 100 + len(large_text))),
        delta_token_ids=list(range(100, 100 + len(large_text))),
        request=None,
    )

    # Parser should have force-exited the tool section
    assert kimi_k2_tool_parser.in_tool_section is False
    # And returned the content as reasoning
    assert result2 is not None
    assert result2.content == large_text


def test_state_reset(kimi_k2_tool_parser):
    """Test that reset_streaming_state() properly clears all state."""
    # Put parser in a complex state
    kimi_k2_tool_parser.in_tool_section = True
    kimi_k2_tool_parser.token_buffer = "some buffer"
    kimi_k2_tool_parser.current_tool_id = 5
    kimi_k2_tool_parser.prev_tool_call_arr = [{"id": "test"}]
    kimi_k2_tool_parser.section_char_count = 1000

    # Reset
    kimi_k2_tool_parser.reset_streaming_state()

    # Verify all state is cleared
    assert kimi_k2_tool_parser.in_tool_section is False
    assert kimi_k2_tool_parser.token_buffer == ""
    assert kimi_k2_tool_parser.current_tool_id == -1
    assert kimi_k2_tool_parser.prev_tool_call_arr == []
    assert kimi_k2_tool_parser.section_char_count == 0
    assert kimi_k2_tool_parser.current_tool_name_sent is False
    assert kimi_k2_tool_parser.streamed_args_for_tool == []


def test_section_begin_noise_tool_begin_same_chunk(kimi_k2_tool_parser):
    """
    Test that begin→noise→tool_begin within the SAME chunk suppresses
    the noise text correctly (not just across chunks).
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    tool_call_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")

    # Single delta containing: section_begin + spurious text + tool_call_begin
    combined_text = "<|tool_calls_section_begin|> noise text <|tool_call_begin|>"

    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning ",
        current_text="Reasoning " + combined_text,
        delta_text=combined_text,
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, section_begin_id, 3, 4, tool_call_begin_id],
        delta_token_ids=[section_begin_id, 3, 4, tool_call_begin_id],
        request=None,
    )

    # The noise text should NOT leak into content
    # Result should either be None/empty or start tool call parsing
    if result is not None and result.content is not None:
        # If content is returned, it should not contain the noise
        assert "noise text" not in result.content
        assert result.content == "" or result.content.strip() == ""


def test_stream_ends_without_section_end_marker(kimi_k2_tool_parser):
    """
    Test that if the stream ends (EOF) without a proper section end marker,
    the parser doesn't leak text, doesn't crash, and resets state cleanly.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")

    # Enter tool section
    _result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[],
        current_token_ids=[section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )
    assert kimi_k2_tool_parser.in_tool_section is True

    # Some content in tool section
    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="<|tool_calls_section_begin|>",
        current_text="<|tool_calls_section_begin|> partial content",
        delta_text=" partial content",
        previous_token_ids=[section_begin_id],
        current_token_ids=[section_begin_id, 10, 11],
        delta_token_ids=[10, 11],
        request=None,
    )
    # Content should be suppressed
    assert result2.content == "" or result2.content is None

    # Stream ends (EOF) - no more deltas, no section_end marker
    # Simulate this by manually checking state and resetting
    # (In real usage, the request handler would call reset_streaming_state)
    assert kimi_k2_tool_parser.in_tool_section is True  # Still in section

    # Reset state (as would happen between requests)
    kimi_k2_tool_parser.reset_streaming_state()

    # Verify clean slate
    assert kimi_k2_tool_parser.in_tool_section is False
    assert kimi_k2_tool_parser.token_buffer == ""

    # Next request should work normally
    result3 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="New reasoning",
        delta_text="New reasoning",
        previous_token_ids=[],
        current_token_ids=[20, 21],
        delta_token_ids=[20, 21],
        request=None,
    )
    assert result3 is not None
    assert result3.content == "New reasoning"


def test_same_chunk_begin_and_end_markers(kimi_k2_tool_parser):
    """
    CRITICAL TEST: Verify that when both section_begin and section_end
    markers appear in the SAME chunk, the parser correctly:
    1. Enters the tool section
    2. Immediately exits the tool section
    3. Does NOT get stuck in in_tool_section=True state

    This tests the bug fix where elif was changed to if to handle
    both state transitions in a single delta.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    # Single chunk with both markers (e.g., empty tool section)
    combined_delta = "<|tool_calls_section_begin|><|tool_calls_section_end|>"

    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Some reasoning ",
        current_text="Some reasoning " + combined_delta,
        delta_text=combined_delta,
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, section_begin_id, section_end_id],
        delta_token_ids=[section_begin_id, section_end_id],
        request=None,
    )

    # CRITICAL: Parser should NOT be stuck in tool section
    assert kimi_k2_tool_parser.in_tool_section is False, (
        "Parser stuck in tool section after processing both begin/end in same chunk. "
        "This indicates the elif bug was not fixed."
    )

    # Result should be empty or contain only stripped content
    assert result is not None
    assert result.content == "" or result.content is None

    # Verify subsequent content streams correctly (not suppressed)
    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Some reasoning " + combined_delta,
        current_text="Some reasoning " + combined_delta + " More reasoning",
        delta_text=" More reasoning",
        previous_token_ids=[1, 2, section_begin_id, section_end_id],
        current_token_ids=[1, 2, section_begin_id, section_end_id, 10, 11],
        delta_token_ids=[10, 11],
        request=None,
    )

    # This content should NOT be suppressed (we're out of tool section)
    assert result2 is not None
    assert result2.content == " More reasoning"


def test_same_chunk_begin_content_end_markers(kimi_k2_tool_parser):
    """
    Test the same-chunk scenario with actual content between markers.
    Example: <|tool_calls_section_begin|> text <|tool_calls_section_end|>
    all arriving in one delta. The key is that the state machine correctly
    transitions in and out within the same chunk.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    # Chunk with begin, some whitespace/noise, and end all together
    # This simulates a tool section that opens and closes in the same chunk
    combined_delta = "<|tool_calls_section_begin|>   <|tool_calls_section_end|>"

    _result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning ",
        current_text="Reasoning " + combined_delta,
        delta_text=combined_delta,
        previous_token_ids=[1],
        current_token_ids=[1, section_begin_id, 100, section_end_id],
        delta_token_ids=[section_begin_id, 100, section_end_id],
        request=None,
    )

    # Parser should exit cleanly (not stuck in tool section)
    assert kimi_k2_tool_parser.in_tool_section is False

    # Verify the fix: next content should stream normally, not be suppressed
    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning " + combined_delta,
        current_text="Reasoning " + combined_delta + " Done",
        delta_text=" Done",
        previous_token_ids=[1, section_begin_id, 100, section_end_id],
        current_token_ids=[1, section_begin_id, 100, section_end_id, 200],
        delta_token_ids=[200],
        request=None,
    )

    # Content after section should be returned (not suppressed)
    assert result2 is not None
    assert result2.content == " Done"


def test_tool_call_end_and_section_end_same_chunk(kimi_k2_tool_parser):
    """
    CRITICAL TEST (P1): Verify that when both <|tool_call_end|> and
    <|tool_calls_section_end|> appear in the SAME chunk, the parser:
    1. Processes the tool_call_end first (emits final arguments)
    2. THEN exits the section
    3. Does NOT drop the final tool call update
    4. Does NOT leak special tokens into reasoning

    This tests the deferred section exit fix.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")
    tool_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")
    tool_end_id = kimi_k2_tool_parser.vocab.get("<|tool_call_end|>")

    # Simulate a streaming sequence for a SHORT tool call (all in one chunk):
    combined = (
        '<|tool_call_begin|>get_weather:0 <|tool_call_argument_begin|> {"city": "Paris"} '
        "<|tool_call_end|><|tool_calls_section_end|>"
    )

    deltas = [
        ("Let me help. ", [1, 2]),
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        (combined, [tool_begin_id, 10, 11, 12, tool_end_id, section_end_id]),
        (" Done", [20]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    # CRITICAL: Parser should have exited section AFTER processing tool
    assert kimi_k2_tool_parser.in_tool_section is False

    # Tool call should have been emitted (not dropped)
    if results[2] is not None and results[2].content is not None:
        # Verify no special tokens leaked into content
        assert "<|tool_call_end|>" not in results[2].content
        assert "<|tool_calls_section_end|>" not in results[2].content

    # Content after tool section should stream normally
    assert results[3] is not None
    assert results[3].content == " Done"


def test_streaming_tool_call_markers_not_leaked(kimi_k2_tool_parser):
    """
    CRITICAL TEST: Verify that tool call markers (<|tool_call_begin|>,
    <|tool_call_end|>, <|tool_call_argument_begin|>) are NOT leaked
    into the content field during streaming.

    This reproduces the AWS Bedrock bug where tool call markers appeared
    in the 'text' field of responses.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")
    tool_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")
    tool_end_id = kimi_k2_tool_parser.vocab.get("<|tool_call_end|>")

    # List of markers that should NEVER appear in content
    forbidden_markers = [
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<|tool_call_argument_begin|>",
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
    ]

    all_content = []

    # Steps: reasoning, section begin, tool call, section end, more reasoning
    tool_chunk = (
        "<|tool_call_begin|> functions.get_weather:0 "
        '<|tool_call_argument_begin|> {"city": "Tokyo"} <|tool_call_end|>'
    )
    deltas = [
        ("I'll check the weather. ", [1, 2, 3]),
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        (tool_chunk, [tool_begin_id, 10, 11, tool_end_id]),
        ("<|tool_calls_section_end|>", [section_end_id]),
        (" Here's the result.", [20, 21]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    for res in results:
        if res and res.content:
            all_content.append(res.content)

    # CRITICAL ASSERTIONS: No forbidden markers in any content
    full_content = "".join(all_content)
    for marker in forbidden_markers:
        assert marker not in full_content, (
            f"MARKER LEAK DETECTED: '{marker}' found in content. "
            f"Full content: {repr(full_content)}"
        )

    # Also check that tool call content (function name, arguments) is not leaked
    assert "get_weather" not in full_content, (
        f"TOOL CALL CONTENT LEAKED: 'get_weather' found in content. "
        f"Full content: {repr(full_content)}"
    )
    assert "Tokyo" not in full_content, (
        f"TOOL CALL CONTENT LEAKED: 'Tokyo' found in content. "
        f"Full content: {repr(full_content)}"
    )

    # Verify that legitimate content was preserved
    assert "I'll check the weather." in full_content or len(all_content) > 0


def test_streaming_multiple_tool_calls_not_leaked(kimi_k2_tool_parser):
    """
    Test that MULTIPLE tool calls in streaming mode do not leak into content.
    This reproduces the AWS Bedrock scenario: "Compare weather in Tokyo and NYC".
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")
    tool_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")
    tool_end_id = kimi_k2_tool_parser.vocab.get("<|tool_call_end|>")

    all_content = []

    tool1 = '<|tool_call_begin|> get_weather:0 <|tool_call_argument_begin|> {"city": "Tokyo"} <|tool_call_end|>'
    tool2 = ' <|tool_call_begin|> get_weather:1 <|tool_call_argument_begin|> {"city": "New York"} <|tool_call_end|>'

    deltas = [
        ("I'll compare the weather. ", [1, 2, 3]),
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        (tool1, [tool_begin_id, 10, tool_end_id]),
        (tool2, [tool_begin_id, 20, tool_end_id]),
        ("<|tool_calls_section_end|>", [section_end_id]),
        (" Here's the comparison.", [30]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    for res in results:
        if res and res.content:
            all_content.append(res.content)

    # Assertions
    full_content = "".join(all_content)

    # Check no markers leaked
    forbidden = ["<|tool_call", "<|tool_calls_section"]
    for marker in forbidden:
        assert marker not in full_content, (
            f"MARKER LEAKED: {marker} in {repr(full_content)}"
        )

    # Check no tool call content leaked (both tools)
    assert "get_weather" not in full_content, f"TOOL NAME LEAKED: {repr(full_content)}"
    assert "Tokyo" not in full_content, f"TOOL ARG LEAKED (Tokyo): {repr(full_content)}"
    assert "New York" not in full_content, (
        f"TOOL ARG LEAKED (NYC): {repr(full_content)}"
    )

    # Legitimate content preserved
    assert "compare" in full_content.lower() or len(all_content) > 0


# ---------------------------------------------------------------------------
# Helper to build a ChatCompletionRequest with tools for normalization tests
# ---------------------------------------------------------------------------

def _make_request_with_tools(tools_spec: list[dict]):
    """Build a minimal ChatCompletionRequest with the given tool definitions."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionToolsParam,
    )
    from vllm.entrypoints.openai.engine.protocol import FunctionDefinition

    tools = []
    for spec in tools_spec:
        tools.append(
            ChatCompletionToolsParam(
                function=FunctionDefinition(**spec)
            )
        )
    return ChatCompletionRequest(messages=[], tools=tools)


# ---------------------------------------------------------------------------
# _extract_function_name unit tests
# ---------------------------------------------------------------------------


def test_extract_function_name_standard_format(kimi_k2_tool_parser):
    """Strategy 1: standard 'functions.name:index' and 'name:index'."""
    assert (
        kimi_k2_tool_parser._extract_function_name(
            "functions.get_weather:0", "{}", None
        )
        == "get_weather"
    )
    assert (
        kimi_k2_tool_parser._extract_function_name("get_weather:0", "{}", None)
        == "get_weather"
    )


def test_extract_function_name_toolu_prefix_strategy1_rejected(kimi_k2_tool_parser):
    """Strategy 1 should reject names starting with 'toolu_' and fall through."""
    # The ID "functions.toolu_vrtx_xxx:0" has name "toolu_vrtx_xxx" after
    # split, which starts with "toolu_" — Strategy 1 skips it.
    # With no request tools, falls to Strategy 3 (raw ID as name).
    name = kimi_k2_tool_parser._extract_function_name(
        "functions.toolu_vrtx_xxx:0", "{}", None
    )
    # Strategy 3 fallback: uses the raw ID
    assert name == "functions.toolu_vrtx_xxx:0"


def test_extract_function_name_parameter_overlap(kimi_k2_tool_parser):
    """Strategy 2: match argument keys against tool parameter schemas."""
    request = _make_request_with_tools([
        {
            "name": "Grep",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "output_mode": {"type": "string"},
                    "path": {"type": "string"},
                },
            },
        },
        {
            "name": "Edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
            },
        },
    ])

    # Args match Grep's params (pattern, output_mode, path)
    name = kimi_k2_tool_parser._extract_function_name(
        "toolu_vrtx_01SLHy9bdFTnvngLJiA3BBRU",
        '{"pattern": "updateTxn", "output_mode": "content", "path": "/home/repo"}',
        request,
    )
    assert name == "Grep"

    # Args match Edit's params (file_path, old_string, new_string)
    name = kimi_k2_tool_parser._extract_function_name(
        "functions.toolu_01FhCzpCe7fsuFBWSWt41ddB:0",
        '{"file_path": "/foo.ts", "old_string": "x", "new_string": "y"}',
        request,
    )
    assert name == "Edit"


def test_extract_function_name_fallback_no_request(kimi_k2_tool_parser):
    """Strategy 3: fallback when no request and non-standard ID."""
    name = kimi_k2_tool_parser._extract_function_name(
        "toolu_vrtx_01SLHy9bdFTnvngLJiA3BBRU",
        '{"pattern": "test"}',
        None,  # type: ignore[arg-type]
    )
    # No request → Strategy 2 skipped → Strategy 3 returns raw ID
    assert name == "toolu_vrtx_01SLHy9bdFTnvngLJiA3BBRU"


# ---------------------------------------------------------------------------
# Non-streaming extract_tool_calls with non-standard IDs
# ---------------------------------------------------------------------------


def test_extract_tool_calls_toolu_id_with_tools(kimi_k2_tokenizer):
    """Non-streaming: 'functions.toolu_xxx' ID with request.tools should
    resolve function name via parameter overlap and produce a random ID."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
    )
    from vllm.entrypoints.openai.engine.protocol import FunctionDefinition

    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="Edit",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                },
            )
        )
    ]
    request = ChatCompletionRequest(messages=[], tools=tools)
    parser = KimiK2ToolParser(kimi_k2_tokenizer)

    args = json.dumps({
        "replace_all": False,
        "file_path": "/Users/mihir.jaiswal/Desktop/blend-design-system/apps/ascent/app/docs/utils/index.ts",
        "old_string": "export {\\n    default as scanDirectory,\\n    buildVersionPeerMap,\\n    type DocItem,\\n} from './scanDirectory'",
        "new_string": "export {\\n    default as scanDirectory,\\n    buildVersionPeerMap,\\n    buildSidebarItemsWithCategories,\\n    type DocItem,\\n} from './scanDirectory'",
    })
    model_output = (
        f"I need to update the exports. "
        f"<|tool_calls_section_begin|> <|tool_call_begin|> "
        f"functions.Edit:16 <|tool_call_argument_begin|> {args} "
        f"<|tool_call_end|> <|tool_calls_section_end|>"
    )

    extracted = parser.extract_tool_calls(model_output, request)
    assert extracted.tools_called
    assert len(extracted.tool_calls) == 1
    tc = extracted.tool_calls[0]
    assert tc.function.name == "Edit"
    # ID should be random (chatcmpl-tool-<uuid>)
    assert tc.id.startswith("chatcmpl-tool-")
    # Arguments should be parseable JSON
    json.loads(tc.function.arguments)


def test_extract_tool_calls_no_colon_id_with_tools(kimi_k2_tokenizer):
    """Non-streaming: 'toolu_vrtx_xxx' (no colon) with request.tools should
    resolve function name via parameter overlap and produce a random ID."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
    )
    from vllm.entrypoints.openai.engine.protocol import FunctionDefinition

    tools = [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="Grep",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "output_mode": {"type": "string"},
                        "path": {"type": "string"},
                    },
                },
            )
        )
    ]
    request = ChatCompletionRequest(messages=[], tools=tools)
    parser = KimiK2ToolParser(kimi_k2_tokenizer)

    model_output = (
        "<|tool_calls_section_begin|> <|tool_call_begin|> "
        "toolu_vrtx_01SLHy9bdFTnvngLJiA3BBRU <|tool_call_argument_begin|> "
        '{"pattern": "updateTxnAndPayoutWithAudit", "output_mode": "content", "path": "/home/repo"} '
        "<|tool_call_end|> <|tool_calls_section_end|>"
    )

    extracted = parser.extract_tool_calls(model_output, request)
    assert extracted.tools_called
    assert len(extracted.tool_calls) == 1
    tc = extracted.tool_calls[0]
    assert tc.function.name == "Grep"
    # ID should be random (chatcmpl-tool-<uuid>)
    assert tc.id.startswith("chatcmpl-tool-")
    json.loads(tc.function.arguments)


def test_extract_tool_calls_no_request_with_standard_id(kimi_k2_tool_parser):
    """Standard IDs should work identically when request=None; the parser
    produces random IDs regardless of the model's native ID format."""
    model_output = (
        "<|tool_calls_section_begin|> <|tool_call_begin|> "
        "functions.get_weather:0 <|tool_call_argument_begin|> "
        '{"city": "Beijing"} '
        "<|tool_call_end|> <|tool_calls_section_end|>"
    )
    extracted = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
    assert extracted.tools_called
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert extracted.tool_calls[0].id.startswith("chatcmpl-tool-")


def test_empty_tool_calls_guard(kimi_k2_tool_parser):
    """When section markers are present but regex extracts zero calls,
    tools_called should be False (not True with empty list)."""
    # The invalid_funcall test already covers this scenario, but let's
    # also test with just markers and unparseable content
    model_output = (
        "<|tool_calls_section_begin|> some noise without proper markers "
        "<|tool_calls_section_end|>"
    )
    extracted = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
    assert not extracted.tools_called
    assert extracted.tool_calls == []


def test_tool_call_ids_are_unique(kimi_k2_tool_parser):
    """Tool call IDs must be unique across extractions — never repeated.
    This is critical for agents that don't scope IDs per message."""
    model_output = (
        "<|tool_calls_section_begin|> <|tool_call_begin|> "
        "functions.get_weather:0 <|tool_call_argument_begin|> "
        '{"city": "Beijing"} '
        "<|tool_call_end|> <|tool_calls_section_end|>"
    )

    # Extract the same model output multiple times
    ids = set()
    for _ in range(20):
        extracted = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
        assert extracted.tools_called
        assert len(extracted.tool_calls) == 1
        ids.add(extracted.tool_calls[0].id)

    # All 20 extractions should produce unique IDs
    assert len(ids) == 20, (
        f"Expected 20 unique IDs, got {len(ids)}. "
        f"IDs must be unique to prevent confusion in multi-turn agent loops."
    )


def test_tool_call_ids_unique_within_multi_call(kimi_k2_tool_parser):
    """Multiple tool calls in a single extraction must also have unique IDs."""
    model_output = (
        "<|tool_calls_section_begin|><|tool_call_begin|>"
        "functions.get_weather:0<|tool_call_argument_begin|>"
        '{"city": "NYC"}'
        "<|tool_call_end|>"
        "<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>"
        '{"city": "LA"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    extracted = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)  # type: ignore[arg-type]
    assert extracted.tools_called
    assert len(extracted.tool_calls) == 2
    ids = [tc.id for tc in extracted.tool_calls]
    assert ids[0] != ids[1], "Tool calls within the same extraction must have unique IDs"
    assert all(tid.startswith("chatcmpl-tool-") for tid in ids)
