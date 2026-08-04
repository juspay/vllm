# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path

from vllm.tokenizers.registry import get_tokenizer
from vllm.v1.sample.token_id_mask import decoded_token_must_be_disallowed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate token IDs whose decoded pieces contain Han characters or "
            "are not valid standalone UTF-8."
        )
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = get_tokenizer(
        args.tokenizer,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    special_ids = set(tokenizer.all_special_ids)
    disallowed_token_ids = []
    for token_id in range(tokenizer.max_token_id + 1):
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded_token_must_be_disallowed(decoded):
            disallowed_token_ids.append(token_id)

    args.output.write_text(
        json.dumps(disallowed_token_ids, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(disallowed_token_ids)} token IDs to {args.output}")


if __name__ == "__main__":
    main()
