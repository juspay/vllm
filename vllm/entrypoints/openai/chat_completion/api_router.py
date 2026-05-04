# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"
LOG_ALL_CHAT_COMPLETION_400_ERRORS = "VLLM_LOG_ALL_CHAT_COMPLETION_400_ERRORS"
CHAT_COMPLETION_400_LOG_KEYWORDS = (
    "expecting value",
    "expecting property name",
    "extra data",
    "invalid control character",
    "invalid \\escape",
    "jsondecodeerror",
    "line 1 column",
    "malformed",
    "tool call",
    "unterminated string",
)


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def batch_chat(request: Request) -> OpenAIServingChatBatch | None:
    return request.app.state.openai_serving_chat_batch


def _get_request_id(raw_request: Request) -> str | None:
    request_metadata = getattr(raw_request.state, "request_metadata", None)
    return getattr(request_metadata, "request_id", None)


def _should_log_chat_completion_400_details(error: ErrorResponse) -> bool:
    if os.environ.get(LOG_ALL_CHAT_COMPLETION_400_ERRORS, "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return True

    error_text = str(error.model_dump()).lower()
    return any(
        keyword in error_text for keyword in CHAT_COMPLETION_400_LOG_KEYWORDS
    )


def _log_chat_error_response(
    request: ChatCompletionRequest, raw_request: Request, error: ErrorResponse
) -> None:
    if not _should_log_chat_completion_400_details(error):
        return

    try:
        logger.error(
            "Chat completion ErrorResponse returned: request_id=%s, "
            "status_code=%s, error=%s, request=%s",
            _get_request_id(raw_request),
            error.error.code,
            error.model_dump(),
            request.model_dump(),
        )
    except Exception:
        logger.exception("Failed to log chat completion ErrorResponse.")


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        _log_chat_error_response(request, raw_request, generator)
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/chat/completions/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_batch_chat_completion(
    request: BatchChatCompletionRequest, raw_request: Request
):
    handler = batch_chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    result = await handler.create_batch_chat_completion(request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)
