#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fs from "node:fs";
import path from "node:path";

const DEFAULT_URL = "https://grid.ai.juspay.net/v1/chat/completions";
const AUTH_ENV_NAMES = [
  "LITELLM_AUTHORIZATION",
  "LITELLM_API_KEY",
  "LITELLM_AUTH",
  "OPENAI_API_KEY",
];

function parseArgs(argv) {
  const args = {
    bodyFile: "kimi-malformed-request.json",
    url: process.env.LITELLM_URL || DEFAULT_URL,
    runs: 1,
    concurrency: 1,
    delayMs: 0,
    timeoutMs: 180_000,
    outDir: "kimi-malformed-repro-output",
    model: process.env.KIMI_REPRO_MODEL || "",
    maxTokens: undefined,
    maxCompletionTokens: undefined,
    seedStart: undefined,
    requestIdPrefix: "kimi-malformed-repro",
    noRequestId: false,
    keepStream: false,
    saveAll: false,
    stopOnMalformed: true,
    validateAssistantJson: true,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) {
        throw new Error(`Missing value for ${arg}`);
      }
      return argv[++i];
    };

    if (arg === "--body-file") args.bodyFile = next();
    else if (arg === "--url") args.url = next();
    else if (arg === "--runs") args.runs = Number(next());
    else if (arg === "--concurrency") args.concurrency = Number(next());
    else if (arg === "--delay-ms") args.delayMs = Number(next());
    else if (arg === "--delay-s") args.delayMs = Number(next()) * 1000;
    else if (arg === "--timeout-ms") args.timeoutMs = Number(next());
    else if (arg === "--timeout-s") args.timeoutMs = Number(next()) * 1000;
    else if (arg === "--out-dir") args.outDir = next();
    else if (arg === "--model") args.model = next();
    else if (arg === "--max-tokens") args.maxTokens = Number(next());
    else if (arg === "--max-completion-tokens") {
      args.maxCompletionTokens = Number(next());
    } else if (arg === "--seed-start") args.seedStart = Number(next());
    else if (arg === "--request-id-prefix") args.requestIdPrefix = next();
    else if (arg === "--no-request-id") args.noRequestId = true;
    else if (arg === "--keep-stream") args.keepStream = true;
    else if (arg === "--save-all") args.saveAll = true;
    else if (arg === "--no-stop-on-malformed") args.stopOnMalformed = false;
    else if (arg === "--stop-on-malformed") args.stopOnMalformed = true;
    else if (arg === "--no-validate-assistant-json") {
      args.validateAssistantJson = false;
    } else if (arg === "--validate-assistant-json") {
      args.validateAssistantJson = true;
    }
    else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!Number.isFinite(args.runs) || args.runs < 1) {
    throw new Error("--runs must be a positive number");
  }
  if (!Number.isFinite(args.concurrency) || args.concurrency < 1) {
    throw new Error("--concurrency must be a positive number");
  }
  args.runs = Math.floor(args.runs);
  args.concurrency = Math.min(Math.floor(args.concurrency), args.runs);
  return args;
}

function printHelp() {
  console.log(`Usage:
  node scripts/kimi_malformed_tool_repro.mjs --body-file request.json [options]

Auth env, first present wins:
  LITELLM_AUTHORIZATION="Bearer ..."
  LITELLM_API_KEY="..."
  LITELLM_AUTH="..."
  OPENAI_API_KEY="..."

Options:
  --url URL                         Default: ${DEFAULT_URL}
  --runs N                          Default: 1
  --concurrency N                   Default: 1
  --out-dir DIR                     Default: kimi-malformed-repro-output
  --model MODEL                     Override request body model
  --request-id-prefix PREFIX        Default: kimi-malformed-repro
  --seed-start N                    Inject seed=N+attempt-1
  --max-tokens N                    Inject max_tokens
  --max-completion-tokens N         Inject max_completion_tokens
  --timeout-s N                     Default: 180
  --delay-s N                       Sleep between attempts per worker
  --save-all                        Save every response
  --no-stop-on-malformed            Continue after malformed/error capture
  --no-validate-assistant-json      Do not validate assistant message.content JSON
  --keep-stream                     Do not force stream=false
`);
}

function readBody(bodyFile) {
  const text = bodyFile === "-"
    ? fs.readFileSync(0, "utf8")
    : fs.readFileSync(bodyFile, "utf8");
  const body = JSON.parse(text);
  if (!body || Array.isArray(body) || typeof body !== "object") {
    throw new Error("request body must be a JSON object");
  }
  return body;
}

function authHeader() {
  for (const envName of AUTH_ENV_NAMES) {
    const value = process.env[envName]?.trim();
    if (!value) continue;
    return value.toLowerCase().startsWith("bearer ") ? value : `Bearer ${value}`;
  }

  throw new Error(`missing auth env; set one of: ${AUTH_ENV_NAMES.join(", ")}`);
}

function jsonTypeName(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (Number.isInteger(value)) return "integer";
  return typeof value === "number" ? "number" : typeof value;
}

function matchesJsonType(value, expected) {
  if (expected === "boolean") return typeof value === "boolean";
  if (expected === "integer") return Number.isInteger(value);
  if (expected === "number") return typeof value === "number";
  if (expected === "string") return typeof value === "string";
  if (expected === "array") return Array.isArray(value);
  if (expected === "object") {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }
  if (expected === "null") return value === null;
  return true;
}

function validateSchemaSubset(value, schema, jsonPath = "$") {
  const errors = [];
  const schemaType = schema?.type;
  const expectedTypes = Array.isArray(schemaType) ? schemaType : [schemaType];
  const filteredTypes = expectedTypes.filter((item) => typeof item === "string");

  if (
    filteredTypes.length > 0 &&
    !filteredTypes.some((expected) => matchesJsonType(value, expected))
  ) {
    errors.push(
      `${jsonPath} expected ${filteredTypes.join("/")} got ${jsonTypeName(value)}`,
    );
    return errors;
  }

  if (Array.isArray(schema?.enum) && !schema.enum.includes(value)) {
    errors.push(`${jsonPath} value ${JSON.stringify(value)} not in enum`);
  }

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return errors;
  }

  if (Array.isArray(schema?.required)) {
    for (const key of schema.required) {
      if (typeof key === "string" && !(key in value)) {
        errors.push(`${jsonPath}.${key} is required but missing`);
      }
    }
  }

  if (schema?.properties && typeof schema.properties === "object") {
    for (const [key, subschema] of Object.entries(schema.properties)) {
      if (!(key in value) || !subschema || typeof subschema !== "object") continue;
      errors.push(...validateSchemaSubset(value[key], subschema, `${jsonPath}.${key}`));
    }
  }

  return errors;
}

function toolSchemasByName(body) {
  const schemas = {};
  for (const tool of body.tools || []) {
    const fn = tool?.function;
    if (!fn || typeof fn !== "object" || typeof fn.name !== "string") continue;
    schemas[fn.name] = fn.parameters && typeof fn.parameters === "object"
      ? fn.parameters
      : {};
  }
  return schemas;
}

function malformedToolCallDetails(responseJson, schemas) {
  const details = [];
  const choices = Array.isArray(responseJson?.choices) ? responseJson.choices : [];

  choices.forEach((choice, fallbackChoiceIndex) => {
    const toolCalls = Array.isArray(choice?.message?.tool_calls)
      ? choice.message.tool_calls
      : [];

    for (const toolCall of toolCalls) {
      const fn = toolCall?.function || {};
      const toolName = fn.name;
      const args = fn.arguments;
      let reason = null;
      let parsedArgs = null;

      if (typeof toolName !== "string") {
        reason = `missing or non-string tool name: ${JSON.stringify(toolName)}`;
      } else if (!(toolName in schemas)) {
        reason = `unknown tool name: ${toolName}`;
      }

      if (reason === null) {
        if (typeof args === "string") {
          try {
            parsedArgs = JSON.parse(args);
          } catch (error) {
            reason = `invalid JSON arguments: ${error.message}`;
          }
        } else if (
          args !== null &&
          typeof args === "object"
        ) {
          parsedArgs = args;
        } else {
          reason = `arguments must be JSON string/object/list, got ${typeof args}`;
        }
      }

      if (reason === null && typeof toolName === "string") {
        const schemaErrors = validateSchemaSubset(parsedArgs, schemas[toolName]);
        if (schemaErrors.length > 0) {
          reason = `schema mismatch: ${schemaErrors.join("; ")}`;
        }
      }

      if (reason !== null) {
        details.push({
          choice_index: choice?.index ?? fallbackChoiceIndex,
          tool_call_id: toolCall?.id,
          tool_name: toolName,
          arguments: args,
          reason,
        });
      }
    }
  });

  return details;
}

function invalidAssistantJsonOutputDetails(responseJson) {
  const details = [];
  const choices = Array.isArray(responseJson?.choices) ? responseJson.choices : [];

  choices.forEach((choice, fallbackChoiceIndex) => {
    const message = choice?.message;
    if (!message || typeof message !== "object") {
      details.push({
        choice_index: choice?.index ?? fallbackChoiceIndex,
        reason: "missing assistant message object",
        content: undefined,
      });
      return;
    }

    const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    if (toolCalls.length > 0) {
      return;
    }

    const content = message.content;
    if (typeof content !== "string" || content.length === 0) {
      details.push({
        choice_index: choice?.index ?? fallbackChoiceIndex,
        reason: `missing assistant content and no tool calls; got ${jsonTypeName(content)}`,
        content,
      });
      return;
    }

    let parsedContent;
    try {
      parsedContent = JSON.parse(content);
    } catch (error) {
      details.push({
        choice_index: choice?.index ?? fallbackChoiceIndex,
        reason: `assistant content is not valid JSON: ${error.message}`,
        content,
      });
      return;
    }

    const schemaErrors = validateAskAiResponseShape(parsedContent);
    if (schemaErrors.length > 0) {
      details.push({
        choice_index: choice?.index ?? fallbackChoiceIndex,
        reason: `assistant JSON shape mismatch: ${schemaErrors.join("; ")}`,
        content,
        parsed_content: parsedContent,
      });
    }
  });

  return details;
}

function validateAskAiResponseShape(value) {
  const errors = [];
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return [`$ expected object got ${jsonTypeName(value)}`];
  }

  const requiredKeys = ["summary", "keypoints", "citations", "userTags"];
  for (const key of requiredKeys) {
    if (!(key in value)) errors.push(`$.${key} is required but missing`);
  }

  if ("summary" in value && typeof value.summary !== "string") {
    errors.push(`$.summary expected string got ${jsonTypeName(value.summary)}`);
  }
  if ("keypoints" in value && !Array.isArray(value.keypoints)) {
    errors.push(`$.keypoints expected array got ${jsonTypeName(value.keypoints)}`);
  }
  if (
    "citations" in value &&
    (value.citations === null ||
      typeof value.citations !== "object" ||
      Array.isArray(value.citations))
  ) {
    errors.push(`$.citations expected object got ${jsonTypeName(value.citations)}`);
  }
  if (
    "userTags" in value &&
    (value.userTags === null ||
      typeof value.userTags !== "object" ||
      Array.isArray(value.userTags))
  ) {
    errors.push(`$.userTags expected object got ${jsonTypeName(value.userTags)}`);
  }

  return errors;
}

function interestingError(status, text) {
  if (status < 400) return false;
  return [
    "Expecting value",
    "JSON",
    "json",
    "tool",
    "BadRequestError",
    "Hosted_vllmException",
  ].some((needle) => text.includes(needle));
}

async function postJson(url, body, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Accept: "*/*",
        Authorization: authHeader(),
        "Content-Type": "application/json",
        "User-Agent": "kimi-malformed-tool-repro/1.0",
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await response.text();
    return {
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      text,
    };
  } finally {
    clearTimeout(timer);
  }
}

function parseJsonOrNull(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function prepareBody(baseBody, args, attempt) {
  const body = structuredClone(baseBody);

  if (args.model) body.model = args.model;
  if (args.maxTokens !== undefined) body.max_tokens = args.maxTokens;
  if (args.maxCompletionTokens !== undefined) {
    body.max_completion_tokens = args.maxCompletionTokens;
  }
  if (!args.keepStream) body.stream = false;
  if (args.seedStart !== undefined) body.seed = args.seedStart + attempt - 1;
  if (!args.noRequestId) {
    body.request_id = `${args.requestIdPrefix}-${String(attempt).padStart(4, "0")}`;
  }

  return body;
}

function writeCapture(outDir, attempt, kind, capture) {
  fs.mkdirSync(outDir, { recursive: true });
  const filePath = path.join(outDir, `${String(attempt).padStart(4, "0")}-${kind}.json`);
  fs.writeFileSync(filePath, `${JSON.stringify(capture, null, 2)}\n`, "utf8");
  return filePath;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runAttempt(attempt, args, baseBody, schemas) {
  const body = prepareBody(baseBody, args, attempt);
  console.log(`[${attempt}/${args.runs}] request_id=${body.request_id ?? ""}`);

  let result;
  try {
    result = await postJson(args.url, body, args.timeoutMs);
  } catch (error) {
    const capture = {
      attempt,
      request_id: body.request_id,
      request: body,
      error_name: error?.name,
      error_message: error?.message,
      error_stack: error?.stack,
      timeout_ms: args.timeoutMs,
    };
    const filePath = writeCapture(args.outDir, attempt, "request-error", capture);
    console.log(
      `  [${attempt}] REQUEST error captured: ${error?.name || "Error"} `
      + `${error?.message || error} ${filePath}`,
    );
    return { attempt, exitCode: 4, captured: true };
  }

  const { status, headers, text } = result;
  const responseJson = parseJsonOrNull(text);
  const details = responseJson ? malformedToolCallDetails(responseJson, schemas) : [];
  const invalidJsonOutputs = (
    responseJson && args.validateAssistantJson
      ? invalidAssistantJsonOutputDetails(responseJson)
      : []
  );
  const invalidResponseJson = status < 400 && responseJson === null;
  const errorIsInteresting = interestingError(status, text);
  const capture = {
    attempt,
    status,
    request_id: body.request_id,
    request: body,
    response_headers: headers,
    response_body: text,
    response_json: responseJson,
    malformed_tool_calls: details,
    invalid_json_outputs: invalidJsonOutputs,
    invalid_response_json: invalidResponseJson,
    interesting_error: errorIsInteresting,
  };

  if (invalidResponseJson) {
    const filePath = writeCapture(args.outDir, attempt, "invalid-response-json", capture);
    console.log(`  [${attempt}] INVALID response JSON captured: ${filePath}`);
    return { attempt, exitCode: 5, captured: true };
  }

  if (details.length > 0) {
    const filePath = writeCapture(args.outDir, attempt, "malformed-tool-call", capture);
    console.log(`  [${attempt}] MALFORMED tool call captured: ${filePath}`);
    for (const detail of details) console.log(`  [${attempt}] - ${detail.reason}`);
    return { attempt, exitCode: 2, captured: true };
  }

  if (invalidJsonOutputs.length > 0) {
    const filePath = writeCapture(args.outDir, attempt, "invalid-json-output", capture);
    console.log(`  [${attempt}] INVALID assistant JSON output captured: ${filePath}`);
    for (const detail of invalidJsonOutputs) {
      console.log(`  [${attempt}] - ${detail.reason}`);
    }
    return { attempt, exitCode: 5, captured: true };
  }

  if (errorIsInteresting) {
    const filePath = writeCapture(args.outDir, attempt, "interesting-error", capture);
    console.log(`  [${attempt}] INTERESTING error captured: status=${status} ${filePath}`);
    return { attempt, exitCode: 3, captured: true };
  }

  if (args.saveAll) {
    const filePath = writeCapture(args.outDir, attempt, "response", capture);
    console.log(`  [${attempt}] saved: status=${status} ${filePath}`);
  } else {
    console.log(`  [${attempt}] ok: status=${status}`);
  }

  return { attempt, exitCode: 0, captured: false };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const baseBody = readBody(args.bodyFile);
  const schemas = toolSchemasByName(baseBody);

  if (Object.keys(schemas).length === 0) {
    console.error("warning: request has no usable tool schemas");
  }

  console.log(`started=${new Date().toISOString()}`);
  console.log(`url=${args.url}`);
  console.log(`body_file=${args.bodyFile}`);
  console.log(`runs=${args.runs}`);
  console.log(`concurrency=${args.concurrency}`);
  console.log(`out_dir=${args.outDir}`);

  let nextAttempt = 1;
  let stopScheduling = false;
  let finalExitCode = 0;

  async function worker() {
    while (!stopScheduling) {
      const attempt = nextAttempt++;
      if (attempt > args.runs) return;

      const result = await runAttempt(attempt, args, baseBody, schemas);
      if (result.exitCode !== 0 && finalExitCode === 0) {
        finalExitCode = result.exitCode;
      }
      if (result.captured && args.stopOnMalformed) {
        stopScheduling = true;
      }

      if (args.delayMs > 0 && !stopScheduling && nextAttempt <= args.runs) {
        await sleep(args.delayMs);
      }
    }
  }

  const workers = Array.from(
    { length: args.concurrency },
    () => worker(),
  );
  await Promise.all(workers);

  if (finalExitCode !== 0) {
    process.exit(finalExitCode);
  }

  console.log("finished without malformed tool calls or interesting errors");
}

main().catch((error) => {
  console.error(error?.stack || error);
  process.exit(1);
});
