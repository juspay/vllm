# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.v1.sample.token_id_mask import (
    DISALLOWED_TOKEN_IDS_CONFIG_KEY,
    DISALLOWED_TOKEN_IDS_FILE_CONFIG_KEY,
    TokenIdLogitsMask,
    decoded_token_must_be_disallowed,
    resolve_disallowed_token_ids,
    validate_token_id_mask_with_spec_decode,
)


def test_resolve_disallowed_token_ids_from_file(tmp_path):
    token_ids_file = tmp_path / "token_ids.json"
    token_ids_file.write_text(json.dumps([7, 2, 7]), encoding="utf-8")

    token_ids = resolve_disallowed_token_ids(
        {DISALLOWED_TOKEN_IDS_FILE_CONFIG_KEY: str(token_ids_file)},
        vocab_size=8,
    )

    assert token_ids == [2, 7]


@pytest.mark.parametrize(
    "token_ids",
    [[True], [1.0], ["1"], [-1], [8]],
)
def test_resolve_disallowed_token_ids_rejects_invalid_values(token_ids):
    with pytest.raises(ValueError):
        resolve_disallowed_token_ids(
            {DISALLOWED_TOKEN_IDS_CONFIG_KEY: token_ids},
            vocab_size=8,
        )


def test_token_id_logits_mask_applies_to_every_row():
    logits = torch.zeros((3, 8))

    TokenIdLogitsMask([2, 7]).apply(logits)

    assert torch.isneginf(logits[:, [2, 7]]).all()
    assert torch.isfinite(logits[:, [0, 1, 3, 4, 5, 6]]).all()


@pytest.mark.parametrize("text", ["English text", "123", "\U0001f642"])
def test_decoded_token_allows_valid_non_han_text(text):
    assert not decoded_token_must_be_disallowed(text)


@pytest.mark.parametrize("text", ["English \u6211", "\ufffd", "prefix\ufffdsuffix"])
def test_decoded_token_disallows_han_and_invalid_utf8(text):
    assert decoded_token_must_be_disallowed(text)


def test_token_id_mask_rejects_synthetic_spec_decode():
    spec_config = type(
        "SpecConfig",
        (),
        {"rejection_sample_method": "synthetic"},
    )()

    with pytest.raises(ValueError, match="synthetic acceptance"):
        validate_token_id_mask_with_spec_decode([2], spec_config)
