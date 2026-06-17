# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Tool-related special tokens that are part of Kimi K2's tool-call format.
# These are expected in model output (even without tools in the request) and
# should be stripped from the *reasoning* channel rather than surfaced to the
# client. (Content-side stripping is handled by the tool parser via
# partial_tag_overlap.)
_TOOL_SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|tool_calls?_section_(?:begin|end)\|>"
    r"|<\|tool_call_(?:begin|end)\|>"
    r"|<\|tool_call_argument_begin\|>"
)


class KimiK2ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Kimi K2 model.

    The Kimi K2 model uses <think>...</think> tokens to denote reasoning text,
    and may implicitly end reasoning by starting a tool call section using
    <|tool_calls_section_begin|>.
    Thinking may also begin without a </think> token.

    Kimi's thinking mode can be disabled via chat_template_kwargs.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        # Check if thinking is disabled via chat_template_kwargs
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = bool(chat_kwargs.get("thinking", True))

        # If thinking is not enabled, use identity parser to fall through
        self._identity_parser: IdentityReasoningParser | None
        if not thinking:
            self._identity_parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._identity_parser = None

        # Token definitions
        self._start_token = "<think>"
        self._end_token = "</think>"
        self._tool_section_start_token = "<|tool_calls_section_begin|>"

        # Alternative end token the model may hallucinate instead of the
        # canonical </think>. When seen, treat it as a reasoning-end marker so
        # content after it is not swallowed into the reasoning block.
        self._alt_end_token = "</thinking>"

        # Get token IDs
        self._start_token_id = self.vocab.get(self._start_token)
        self._end_token_id = self.vocab.get(self._end_token)
        self._tool_section_start_token_id = self.vocab.get(
            self._tool_section_start_token
        )
        self._alt_end_token_id = self.vocab.get(self._alt_end_token)

        if self._start_token_id is None or self._end_token_id is None:
            raise RuntimeError(
                "KimiK2ReasoningParser could not locate think start/end "
                "tokens in the tokenizer!"
            )

    @staticmethod
    def _strip_tool_tokens(text: str | None) -> str | None:
        """Strip Kimi tool-call special tokens from a reasoning string."""
        if not text:
            return text
        return _TOOL_SPECIAL_TOKEN_PATTERN.sub("", text)

    @property
    def reasoning_start_str(self) -> str | None:
        return self._start_token

    @property
    def reasoning_end_str(self) -> str | None:
        return self._end_token

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        Reasoning ends when we see either:
        1. The end token (</think>)
        2. The tool section start token (<|tool_calls_section_begin|>)
        """
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end(input_ids)

        start_token_id = self._start_token_id
        end_token_id = self._end_token_id
        tool_section_start_token_id = self._tool_section_start_token_id
        alt_end_token_id = self._alt_end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == end_token_id:
                return True
            # Alternative end token (</thinking>) the model may hallucinate
            if alt_end_token_id is not None and input_ids[i] == alt_end_token_id:
                return True
            # Implicit reasoning end via tool call section
            if (
                tool_section_start_token_id is not None
                and input_ids[i] == tool_section_start_token_id
            ):
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """
        Check if the reasoning content ends in the input_ids on a decode step.
        """
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end_streaming(
                input_ids, delta_ids
            )

        # Materialize iterable for membership checks
        delta_ids_set = set(delta_ids)

        # Check for explicit end token or implicit tool section start in delta
        if self._end_token_id in delta_ids_set:
            return True
        # Alternative end token (</thinking>)
        if (
            self._alt_end_token_id is not None
            and self._alt_end_token_id in delta_ids_set
        ):
            return True
        return (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in delta_ids_set
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        if self._identity_parser is not None:
            return self._identity_parser.extract_content_ids(input_ids)

        if self._end_token_id in input_ids:
            end_token_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._end_token_id)
            )

            if end_token_index != -1:
                return input_ids[end_token_index + 1 :]

        # Alternative end token (</thinking>)
        if (
            self._alt_end_token_id is not None
            and self._alt_end_token_id in input_ids
        ):
            alt_end_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._alt_end_token_id)
            )

            if alt_end_index != -1:
                return input_ids[alt_end_index + 1 :]

        if (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in input_ids
        ):
            tool_section_index = (
                len(input_ids)
                - 1
                - input_ids[::-1].index(self._tool_section_start_token_id)
            )

            if tool_section_index != -1:
                return input_ids[tool_section_index:]

        # still reasoning (no content)
        return []

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
        """
        if self._identity_parser is not None:
            return self._identity_parser.extract_reasoning(model_output, request)

        # thinking does not require a think start token but consume it if present
        start_token_index = model_output.find(self._start_token)
        start_token_index = 0 if start_token_index != 0 else len(self._start_token)
        end_token_index = model_output.find(self._end_token)

        if end_token_index != -1:
            return (
                self._strip_tool_tokens(model_output[start_token_index:end_token_index]),
                model_output[end_token_index + len(self._end_token) :] or None,
            )

        # Alternative end token (</thinking>) the model may hallucinate
        alt_end_index = model_output.find(self._alt_end_token)
        if alt_end_index != -1:
            return (
                self._strip_tool_tokens(model_output[start_token_index:alt_end_index]),
                model_output[alt_end_index + len(self._alt_end_token) :] or None,
            )

        tool_section_index = model_output.find(self._tool_section_start_token)
        if tool_section_index != -1:
            # Keep the tool-section start token in content so the tool parser's
            # extract_tool_calls() can detect tool calls via text matching.
            return (
                self._strip_tool_tokens(
                    model_output[start_token_index:tool_section_index]
                ),
                model_output[tool_section_index:] or None,
            )

        # still reasoning (no content)
        return (
            self._strip_tool_tokens(model_output[start_token_index:]),
            None,
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message during streaming.
        """
        if self._identity_parser is not None:
            return self._identity_parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        # If reasoning has already ended in previous tokens, this is content
        if self.is_reasoning_end(previous_token_ids):
            return DeltaMessage(content=delta_text)

        # Skip single special tokens
        skip_token_ids = [self._start_token_id, self._end_token_id]
        if self._alt_end_token_id is not None:
            skip_token_ids.append(self._alt_end_token_id)
        if len(delta_token_ids) == 1 and delta_token_ids[0] in skip_token_ids:
            return None

        if self._end_token_id in delta_token_ids:
            if self._end_token not in delta_text:
                # Token ID arrived before text was flushed (stop-sequence buffering).
                # Wait for the next delta when the text becomes visible.
                return None
            end_index = delta_text.find(self._end_token)
            reasoning = self._strip_tool_tokens(delta_text[:end_index])
            content = delta_text[end_index + len(self._end_token) :]
            return DeltaMessage(
                reasoning=reasoning, content=content if content else None
            )

        # Alternative end token (</thinking>) in delta
        if (
            self._alt_end_token_id is not None
            and self._alt_end_token_id in delta_token_ids
        ):
            if self._alt_end_token not in delta_text:
                # Token ID arrived before text was flushed (stop-sequence buffering).
                return None
            alt_end_index = delta_text.find(self._alt_end_token)
            reasoning = self._strip_tool_tokens(delta_text[:alt_end_index])
            content = delta_text[alt_end_index + len(self._alt_end_token) :]
            return DeltaMessage(
                reasoning=reasoning, content=content if content else None
            )

        if self._tool_section_start_token_id in delta_token_ids:
            if self._tool_section_start_token not in delta_text:
                # Token ID arrived before text was flushed (stop-sequence buffering).
                return None
            tool_index = delta_text.find(self._tool_section_start_token)
            reasoning = self._strip_tool_tokens(delta_text[:tool_index])
            content = delta_text[tool_index:]
            return DeltaMessage(reasoning=reasoning, content=content)

        # still reasoning (no end token)
        return DeltaMessage(reasoning=self._strip_tool_tokens(delta_text))

    def get_streaming_fallback_content(
        self, text: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> str | None:
        """Promote accumulated reasoning into content on the terminal streaming
        delta when reasoning never ended (model produced no </think>,
        </thinking>, or tool section), so OpenAI clients don't receive null
        content. Invoked by DelegatingParser.finalize_generation only when
        ``state.reasoning_ended`` is False.
        """
        if self._identity_parser is not None:
            return None
        reasoning, content = self.extract_reasoning(text, request)
        if content is None and reasoning:
            return reasoning
        return None
