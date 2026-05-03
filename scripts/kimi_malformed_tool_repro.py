#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a chat-completions request and capture malformed tool calls.

The request body is intentionally read from a JSON file so large production
payloads can be replayed without fragile shell quoting.
"""

import argparse
import copy
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://grid.ai.juspay.net/v1/chat/completions"
AUTH_ENV_NAMES = (
    "LITELLM_AUTHORIZATION",
    "LITELLM_API_KEY",
    "LITELLM_AUTH",
    "OPENAI_API_KEY",
)


def json_type_name(value: Any) -> str:
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


def matches_json_type(value: Any, expected: str) -> bool:
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


def validate_schema_subset(value: Any,
                           schema: Mapping[str, Any],
                           path: str = "$") -> list[str]:
    errors: list[str] = []

    expected_type = schema.get("type")
    expected_types = expected_type if isinstance(expected_type, list) else [
        expected_type
    ]
    expected_types = [t for t in expected_types if isinstance(t, str)]
    if expected_types and not any(
            matches_json_type(value, expected) for expected in expected_types):
        errors.append(
            f"{path} expected {'/'.join(expected_types)} got "
            f"{json_type_name(value)}")
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
            errors.extend(validate_schema_subset(value[key], subschema,
                                                 f"{path}.{key}"))

    return errors


def load_body(path: str) -> dict[str, Any]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def auth_header() -> str:
    for env_name in AUTH_ENV_NAMES:
        value = os.environ.get(env_name)
        if value:
            value = value.strip()
            return value if value.lower().startswith("bearer ") else f"Bearer {value}"

    names = ", ".join(AUTH_ENV_NAMES)
    raise RuntimeError(f"missing auth env; set one of: {names}")


def tool_schemas_by_name(body: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        parameters = function.get("parameters")
        schemas[name] = parameters if isinstance(parameters, dict) else {}
    return schemas


def malformed_tool_call_details(
        response_json: Mapping[str, Any],
        schemas: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []

    choices = response_json.get("choices") or []
    if not isinstance(choices, list):
        return details

    for choice_index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            if not isinstance(function, dict):
                continue

            tool_name = function.get("name")
            arguments = function.get("arguments")
            reason = None
            parsed_args = None

            if not isinstance(tool_name, str):
                reason = f"missing or non-string tool name: {tool_name!r}"
            elif tool_name not in schemas:
                reason = f"unknown tool name: {tool_name}"

            if reason is None:
                if isinstance(arguments, str):
                    try:
                        parsed_args = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        reason = f"invalid JSON arguments: {exc}"
                elif isinstance(arguments, (dict, list)):
                    parsed_args = arguments
                else:
                    reason = (
                        "arguments must be JSON string/object/list, got "
                        f"{type(arguments).__name__}")

            if reason is None and isinstance(tool_name, str):
                schema_errors = validate_schema_subset(parsed_args,
                                                       schemas[tool_name])
                if schema_errors:
                    reason = "schema mismatch: " + "; ".join(schema_errors)

            if reason is not None:
                details.append({
                    "choice_index": choice.get("index", choice_index),
                    "tool_call_id": tool_call.get("id"),
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "reason": reason,
                })

    return details


def post_json(url: str, body: Mapping[str, Any],
              timeout_s: float) -> tuple[int, dict[str, str], str]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "*/*",
            "Authorization": auth_header(),
            "Content-Type": "application/json",
            "User-Agent": "kimi-malformed-tool-repro/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return (
                response.status,
                dict(response.headers.items()),
                response.read().decode("utf-8", errors="replace"),
            )
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            dict(exc.headers.items()),
            exc.read().decode("utf-8", errors="replace"),
        )


def parse_response(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def is_error_interesting(status: int, text: str) -> bool:
    if status < 400:
        return False
    needles = (
        "Expecting value",
        "JSON",
        "json",
        "tool",
        "BadRequestError",
        "Hosted_vllmException",
    )
    return any(needle in text for needle in needles)


def write_capture(out_dir: Path, attempt: int, kind: str,
                  capture: Mapping[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{attempt:04d}-{kind}.json"
    path.write_text(
        json.dumps(capture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def prepare_body(base_body: Mapping[str, Any], args: argparse.Namespace,
                 attempt: int) -> dict[str, Any]:
    body = copy.deepcopy(base_body)

    if args.model:
        body["model"] = args.model
    if args.max_tokens is not None:
        body["max_tokens"] = args.max_tokens
    if args.max_completion_tokens is not None:
        body["max_completion_tokens"] = args.max_completion_tokens
    if not args.keep_stream:
        body["stream"] = False
    if args.seed_start is not None:
        body["seed"] = args.seed_start + attempt - 1
    if not args.no_request_id:
        body["request_id"] = f"{args.request_id_prefix}-{attempt:04d}"

    return body


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a LiteLLM/vLLM chat-completions request and "
        "capture malformed tool-call JSON/schema cases.")
    parser.add_argument("--body-file",
                        default="kimi-malformed-request.json",
                        help="JSON request body file, or '-' for stdin.")
    parser.add_argument("--url",
                        default=os.environ.get("LITELLM_URL", DEFAULT_URL),
                        help="Chat completions URL.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--delay-s", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--out-dir",
                        default="kimi-malformed-repro-output",
                        help="Directory for captured request/response JSON.")
    parser.add_argument("--model",
                        default=os.environ.get("KIMI_REPRO_MODEL"),
                        help="Override request body model.")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--seed-start",
                        type=int,
                        default=None,
                        help="If set, inject seed_start + attempt - 1.")
    parser.add_argument("--request-id-prefix",
                        default="kimi-malformed-repro",
                        help="Prefix for injected request_id.")
    parser.add_argument("--no-request-id",
                        action="store_true",
                        help="Do not inject a per-attempt request_id.")
    parser.add_argument("--keep-stream",
                        action="store_true",
                        help="Do not force stream=false.")
    parser.add_argument("--save-all",
                        action="store_true",
                        help="Save every response, not only malformed cases.")
    parser.add_argument("--stop-on-malformed",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Stop after the first malformed or interesting "
                        "error capture.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_body = load_body(args.body_file)
    schemas = tool_schemas_by_name(base_body)
    out_dir = Path(args.out_dir)

    if not schemas:
        print("warning: request has no usable tool schemas", file=sys.stderr)

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"started={started}")
    print(f"url={args.url}")
    print(f"body_file={args.body_file}")
    print(f"runs={args.runs}")
    print(f"out_dir={out_dir}")

    for attempt in range(1, args.runs + 1):
        body = prepare_body(base_body, args, attempt)
        request_id = body.get("request_id")
        print(f"[{attempt}/{args.runs}] request_id={request_id}")

        status, headers, text = post_json(args.url, body, args.timeout_s)
        response_json = parse_response(text)
        details = (
            malformed_tool_call_details(response_json, schemas)
            if isinstance(response_json, dict) else [])
        interesting_error = is_error_interesting(status, text)

        capture = {
            "attempt": attempt,
            "status": status,
            "request_id": request_id,
            "request": body,
            "response_headers": headers,
            "response_body": text,
            "response_json": response_json,
            "malformed_tool_calls": details,
            "interesting_error": interesting_error,
        }

        if details:
            path = write_capture(out_dir, attempt, "malformed-tool-call",
                                 capture)
            print(f"  MALFORMED tool call captured: {path}")
            for detail in details:
                print(f"  - {detail['reason']}")
            if args.stop_on_malformed:
                return 2
        elif interesting_error:
            path = write_capture(out_dir, attempt, "interesting-error", capture)
            print(f"  INTERESTING error captured: status={status} {path}")
            if args.stop_on_malformed:
                return 3
        elif args.save_all:
            path = write_capture(out_dir, attempt, "response", capture)
            print(f"  saved: status={status} {path}")
        else:
            print(f"  ok: status={status}")

        if args.delay_s and attempt != args.runs:
            time.sleep(args.delay_s)

    print("finished without malformed tool calls or interesting errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
