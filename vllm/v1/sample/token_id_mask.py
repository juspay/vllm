# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

DISALLOWED_TOKEN_IDS_CONFIG_KEY = "disallowed_token_ids"
DISALLOWED_TOKEN_IDS_FILE_CONFIG_KEY = "disallowed_token_ids_file"
INVALID_UTF8_REPLACEMENT_CHARACTER = "\ufffd"
HAN_CODEPOINT_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x2EBF0, 0x2EE5F),
    (0x30000, 0x3134F),
    (0x31350, 0x323AF),
)


def decoded_token_must_be_disallowed(text: str) -> bool:
    # Invalid standalone UTF-8 pieces could otherwise compose a Han character
    # across token boundaries in byte-level tokenizers.
    if INVALID_UTF8_REPLACEMENT_CHARACTER in text:
        return True
    return any(
        start <= ord(character) <= end
        for character in text
        for start, end in HAN_CODEPOINT_RANGES
    )


def resolve_disallowed_token_ids(
    additional_config: dict[str, Any] | Any,
    vocab_size: int,
) -> list[int]:
    if not isinstance(additional_config, dict):
        return []

    inline_ids = additional_config.get(DISALLOWED_TOKEN_IDS_CONFIG_KEY)
    token_ids_file = additional_config.get(DISALLOWED_TOKEN_IDS_FILE_CONFIG_KEY)
    if inline_ids is not None and token_ids_file is not None:
        raise ValueError(
            f"Specify only one of {DISALLOWED_TOKEN_IDS_CONFIG_KEY!r} and "
            f"{DISALLOWED_TOKEN_IDS_FILE_CONFIG_KEY!r}."
        )

    source = DISALLOWED_TOKEN_IDS_CONFIG_KEY
    token_ids = inline_ids
    if token_ids_file is not None:
        if not isinstance(token_ids_file, str) or not token_ids_file:
            raise ValueError(
                f"{DISALLOWED_TOKEN_IDS_FILE_CONFIG_KEY!r} must be a non-empty path."
            )
        source = token_ids_file
        try:
            with Path(token_ids_file).open(encoding="utf-8") as file:
                token_ids = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Unable to load disallowed token IDs from {token_ids_file!r}."
            ) from error

    if token_ids is None:
        return []
    if not isinstance(token_ids, list):
        raise ValueError(f"Disallowed token IDs from {source!r} must be a list.")

    invalid_types = [token_id for token_id in token_ids if type(token_id) is not int]
    if invalid_types:
        raise ValueError(
            f"Disallowed token IDs from {source!r} must contain only integers."
        )

    invalid_ids = [
        token_id for token_id in token_ids if token_id < 0 or token_id >= vocab_size
    ]
    if invalid_ids:
        raise ValueError(
            f"Disallowed token IDs from {source!r} must be in [0, {vocab_size}). "
            f"Invalid IDs: {invalid_ids[:10]}"
        )

    return sorted(set(token_ids))


def validate_token_id_mask_with_spec_decode(
    token_ids: list[int],
    spec_config: SpeculativeConfig | None,
) -> None:
    if (
        token_ids
        and spec_config is not None
        and spec_config.rejection_sample_method == "synthetic"
    ):
        raise ValueError(
            "Disallowed token IDs are incompatible with synthetic rejection "
            "sampling because synthetic acceptance can bypass target logits. "
            "Use standard or block rejection sampling."
        )


class TokenIdLogitsMask:
    def __init__(
        self,
        token_ids: list[int] | None,
        device: torch.device | str | None = None,
    ) -> None:
        self.token_ids = torch.tensor(
            token_ids or [],
            dtype=torch.long,
            device=device,
        )

    @property
    def enabled(self) -> bool:
        return self.token_ids.numel() > 0

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            logits.index_fill_(-1, self.token_ids, -float("inf"))
        return logits
