# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.reasoning.kimi_k2_reasoning_parser import KimiK2ReasoningParser
from vllm.tokenizers import get_tokenizer

REASONING_MODEL_NAME = "moonshotai/Kimi-K2.5"


@pytest.fixture
def mock_kimi_k2_tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<think>": 100,
        "</think>": 101,
        "</thinking>": 102,
        "<|tool_calls_section_begin|>": 200,
        "<|tool_calls_section_end|>": 201,
        "<|tool_call_begin|>": 202,
        "<|tool_call_end|>": 203,
    }
    return tokenizer


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return get_tokenizer(tokenizer_name=REASONING_MODEL_NAME, trust_remote_code=True)


def test_parser_selection_thinking_enabled(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(
        kimi_k2_tokenizer, chat_template_kwargs={"thinking": True}
    )
    assert parser._identity_parser is None


def test_parser_selection_thinking_disabled(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(
        kimi_k2_tokenizer, chat_template_kwargs={"thinking": False}
    )
    assert isinstance(parser._identity_parser, IdentityReasoningParser)


def test_extract_reasoning_with_think_tags(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think>step by step reasoning</think>final answer", request
    )
    assert reasoning == "step by step reasoning"
    assert content == "final answer"


def test_extract_reasoning_empty_thinking(kimi_k2_tokenizer):
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think></think>final answer", request
    )
    assert reasoning == ""
    assert content == "final answer"


def test_extract_reasoning_implicit_start(kimi_k2_tokenizer):
    """When there's no <think> tag, everything is treated as reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "implicit reasoning with no tags", request
    )
    assert reasoning == "implicit reasoning with no tags"
    assert content is None


def test_extract_reasoning_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> implicitly ends reasoning."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    text = "some reasoning<|tool_calls_section_begin|>tool call data"
    reasoning, content = parser.extract_reasoning(text, request)
    assert reasoning == "some reasoning"
    assert content == "<|tool_calls_section_begin|>tool call data"


def test_streaming_reasoning_then_content(kimi_k2_tokenizer):
    """Token-by-token streaming: reasoning tokens then content after </think>."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)

    think_id = parser._start_token_id
    end_think_id = parser._end_token_id
    # Use a real token ID from the tokenizer for regular content
    regular_id = kimi_k2_tokenizer.encode("hello", add_special_tokens=False)[0]

    # First token: <think> — single special token should be skipped
    result = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<think>",
        delta_text="<think>",
        previous_token_ids=[],
        current_token_ids=[think_id],
        delta_token_ids=[think_id],
    )
    assert result is None

    # Reasoning token
    result = parser.extract_reasoning_streaming(
        previous_text="<think>",
        current_text="<think>step one",
        delta_text="step one",
        previous_token_ids=[think_id],
        current_token_ids=[think_id, regular_id],
        delta_token_ids=[regular_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.reasoning == "step one"
    assert result.content is None

    # End token </think> as single token — should be skipped
    result = parser.extract_reasoning_streaming(
        previous_text="<think>step one",
        current_text="<think>step one</think>",
        delta_text="</think>",
        previous_token_ids=[think_id, regular_id],
        current_token_ids=[think_id, regular_id, end_think_id],
        delta_token_ids=[end_think_id],
    )
    assert result is None

    # Content after </think>
    content_id = kimi_k2_tokenizer.encode("world", add_special_tokens=False)[0]
    result = parser.extract_reasoning_streaming(
        previous_text="<think>step one</think>",
        current_text="<think>step one</think>answer",
        delta_text="answer",
        previous_token_ids=[think_id, regular_id, end_think_id],
        current_token_ids=[think_id, regular_id, end_think_id, content_id],
        delta_token_ids=[content_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "answer"


def test_streaming_tool_section_ends_reasoning(kimi_k2_tokenizer):
    """<|tool_calls_section_begin|> in delta ends reasoning during streaming."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)

    think_id = parser._start_token_id
    tool_begin_id = parser._tool_section_start_token_id
    regular_id = kimi_k2_tokenizer.encode("hello", add_special_tokens=False)[0]

    # Tool section token arrives — should transition from reasoning to content
    result = parser.extract_reasoning_streaming(
        previous_text="<think>thinking",
        current_text="<think>thinking<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[think_id, regular_id],
        current_token_ids=[think_id, regular_id, tool_begin_id],
        delta_token_ids=[tool_begin_id],
    )
    assert isinstance(result, DeltaMessage)
    assert result.content == "<|tool_calls_section_begin|>"


# --- Juspay custom fixes: alt-end token, tool-token strip, fallback hook ---


def test_extract_reasoning_alt_end_token(kimi_k2_tokenizer):
    """The model may hallucinate </thinking> instead of </think>."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think>reasoning here</thinking>final answer", request
    )
    assert reasoning == "reasoning here"
    assert content == "final answer"


def test_extract_reasoning_strips_tool_tokens_from_reasoning(kimi_k2_tokenizer):
    """Tool-call special tokens must not leak into the reasoning channel."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    reasoning, content = parser.extract_reasoning(
        "<think>thinking<|tool_call_begin|>oops</think>answer", request
    )
    assert "<|tool_call_begin|>" not in reasoning
    assert reasoning == "thinkingoops"
    assert content == "answer"


def test_get_streaming_fallback_content_promotes_reasoning(kimi_k2_tokenizer):
    """When the model never closes </think>, the whole output is reasoning;
    the fallback hook promotes it to content so clients get non-null content."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    promoted = parser.get_streaming_fallback_content(
        "<think>everything is reasoning, no close tag", request
    )
    assert promoted == "everything is reasoning, no close tag"


def test_get_streaming_fallback_content_none_when_content_present(kimi_k2_tokenizer):
    """If reasoning ended (content exists), the fallback must not fire."""
    parser = KimiK2ReasoningParser(kimi_k2_tokenizer)
    request = ChatCompletionRequest(model="test-model", messages=[], temperature=1.0)

    promoted = parser.get_streaming_fallback_content(
        "<think>reasoning</think>real content", request
    )
    assert promoted is None


def test_is_reasoning_end_alt_token(mock_kimi_k2_tokenizer):
    """</thinking> token id ends reasoning."""
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    think_id = parser._start_token_id
    alt_end_id = parser._alt_end_token_id
    assert alt_end_id is not None
    assert parser.is_reasoning_end([think_id, 999, alt_end_id]) is True
    assert parser.is_reasoning_end([think_id, 999]) is False


def test_streaming_alt_end_token(mock_kimi_k2_tokenizer):
    """</thinking> in a streaming delta splits reasoning from content."""
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    think_id = parser._start_token_id
    alt_end_id = parser._alt_end_token_id

    result = parser.extract_reasoning_streaming(
        previous_text="<think>thinking",
        current_text="<think>thinking</thinking>ans",
        delta_text="</thinking>ans",
        previous_token_ids=[think_id, 999],
        current_token_ids=[think_id, 999, alt_end_id, 998],
        delta_token_ids=[alt_end_id, 998],
    )
    assert isinstance(result, DeltaMessage)
    assert result.reasoning == ""
    assert result.content == "ans"


def test_streaming_end_token_id_buffered(mock_kimi_k2_tokenizer):
    """When stop sequences buffer text, </think> ID arrives before its text.

    The token ID is present in delta_token_ids but the actual string is not
    yet in delta_text (still buffered). The parser must return None to wait
    for the next delta, instead of calling find() which returns -1 and
    silently corrupting the text split.
    """
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    think_id = parser._start_token_id
    end_think_id = parser._end_token_id

    # Simulate: </think> ID arrived but text not yet flushed.
    # Two token IDs in delta to bypass the single-special-token guard.
    result = parser.extract_reasoning_streaming(
        previous_text="some reasoning",
        current_text="some reasoning extra",
        delta_text="extra",  # </think> text not yet flushed
        previous_token_ids=[think_id],
        current_token_ids=[think_id, end_think_id, 999],
        delta_token_ids=[end_think_id, 999],
    )
    assert result is None


def test_streaming_tool_section_id_buffered(mock_kimi_k2_tokenizer):
    """When stop sequences buffer text, tool section start ID arrives before its text.

    Same buffering scenario as above but for <|tool_calls_section_begin|>.
    Without the guard, find() returns -1 and delta_text[:tool_index] silently
    drops the last character of reasoning.
    """
    parser = KimiK2ReasoningParser(mock_kimi_k2_tokenizer)
    think_id = parser._start_token_id
    tool_begin_id = parser._tool_section_start_token_id

    result = parser.extract_reasoning_streaming(
        previous_text="some reasoning",
        current_text="some reasoning extra",
        delta_text="extra",  # tool section text not yet flushed
        previous_token_ids=[think_id],
        current_token_ids=[think_id, tool_begin_id, 999],
        delta_token_ids=[tool_begin_id, 999],
    )
    assert result is None
