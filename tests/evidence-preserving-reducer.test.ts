/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

import { randomUUID } from "node:crypto";
import { mkdtemp, readFile, rm, stat, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { isAbsolute, join, relative } from "node:path";
import type { AssistantMessage, Context, Model } from "@earendil-works/pi-ai";
import type { ExtensionContext, ToolResultEvent } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
	createEvidencePreservingReducerExtension,
	DIAGNOSTIC_COMMAND,
	loadReducerConfig,
	REDUCER_RECEIPT_SCHEMA,
	validateReceipt,
} from "../src/sol-pi/extensions/evidence-preserving-reducer/index.ts";
import { archiveBody } from "../src/sol-pi/extensions/evidence-preserving-reducer/archive.ts";
import {
	callReducer,
	type CompatComplete,
} from "../src/sol-pi/extensions/evidence-preserving-reducer/provider.ts";
import { runtimeRoot } from "../src/sol-pi/runtime-paths.ts";
import { FakePi, FakeSessionManager, fakeContext } from "./helpers.ts";

const cleanupPaths: string[] = [];

const ACTIVE_MODEL = {
	id: ["gpt-5.6", "sol"].join("-"),
	name: "GPT-5.6 SoL",
	api: "openai-responses",
	provider: "openai-codex",
	baseUrl: "https://example.invalid/v1",
	reasoning: true,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 200_000,
	maxTokens: 16_384,
} satisfies Model<"openai-responses">;

const REDUCER_MODEL = {
	id: ["gpt-5.6", "luna"].join("-"),
	name: "GPT-5.6 Luna",
	api: "openai-responses",
	provider: "openai-codex",
	baseUrl: "https://example.invalid/v1",
	reasoning: true,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 32_768,
	maxTokens: 4_096,
} satisfies Model<"openai-responses">;

type Complete = (
	model: Model<string>,
	context: Context,
	options?: Record<string, unknown>,
) => Promise<AssistantMessage>;

interface ModelReceipt {
	schema: string;
	source_sha256: string;
	status: "success" | "failure";
	uncertain: boolean;
	evidence: { kind: string; quote: string }[];
}

interface CapturedCall {
	readonly context: Context;
	readonly model: Model<string>;
	readonly options: Record<string, unknown>;
}

afterEach(async () => {
	vi.useRealTimers();
	await Promise.all(cleanupPaths.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

async function storeRoot(): Promise<string> {
	const value = await mkdtemp(join(tmpdir(), "evidence-preserving-reducer-test-"));
	cleanupPaths.push(value);
	return value;
}

function bashEvent(body: string, overrides: Partial<ToolResultEvent> = {}): ToolResultEvent {
	return {
		type: "tool_result",
		toolName: "bash",
		toolCallId: "call-1",
		input: { command: "pytest -q" },
		content: [{ type: "text", text: body }],
		details: undefined,
		isError: true,
		...overrides,
	} as ToolResultEvent;
}

function fusedEvent(body: string, failed: boolean): ToolResultEvent {
	const marker = failed ? "[then_run:failed]" : "[then_run:succeeded]";
	const confirmation = "Successfully wrote 12 bytes to target.ts";
	return {
		type: "tool_result",
		toolName: "write",
		toolCallId: "write-1",
		input: { path: "target.ts", content: "export {};\n", then_run: { command: "npm test" } },
		content: failed
			? [{ type: "text", text: `${confirmation}\n\n${marker}\n\n${body}` }]
			: [
					{ type: "text", text: confirmation },
					{ type: "text", text: `${marker}\n${body}` },
				],
		details: { patch: "test patch" },
		isError: failed,
	} as ToolResultEvent;
}

function contextInput(context: Context): string {
	const message = context.messages[0];
	if (message?.role !== "user") throw new Error("reducer request omitted its user message");
	if (typeof message.content === "string") return message.content;
	return message.content.flatMap((item) => (item.type === "text" ? [item.text] : [])).join("\n");
}

function sourceHash(input: string): string {
	const match = input.match(/source_sha256=([a-f0-9]{64})/u);
	if (!match?.[1]) throw new Error("request omitted source hash");
	return match[1];
}

function modelComplete(
	body: string,
	receiptFactory: (input: string) => ModelReceipt,
	stopReason: AssistantMessage["stopReason"] = "stop",
	onCall?: (call: CapturedCall) => void,
): Complete {
	return async (model, context, options = {}) => {
		onCall?.({ model, context, options });
		const input = contextInput(context);
		const receipt = receiptFactory(input);
		return {
			role: "assistant",
			content: stopReason === "error" ? [] : [{ type: "text", text: JSON.stringify(receipt) }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: Math.ceil(body.length / 4),
				output: 90,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: Math.ceil(body.length / 4) + 90,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason,
			...(stopReason === "error" ? { errorMessage: "model call failed" } : {}),
			timestamp: Date.now(),
		};
	};
}

function load(
	root: string,
	complete: Complete,
	model: Model<string> | null = ACTIVE_MODEL,
	overrides: Partial<ExtensionContext> = {},
): { context: ExtensionContext; manager: FakeSessionManager; pi: FakePi } {
	const manager = new FakeSessionManager([], "reducer", root);
	const pi = new FakePi(manager);
	createEvidencePreservingReducerExtension()(pi.asExtensionApi());
	const context = fakeContext(manager, {
		model: model ?? undefined,
		modelRegistry: {
			find: (provider: string, modelId: string) =>
				provider === REDUCER_MODEL.provider && modelId === REDUCER_MODEL.id ? REDUCER_MODEL : undefined,
			complete,
		} as unknown as ExtensionContext["modelRegistry"],
		...overrides,
	});
	return { context, manager, pi };
}

describe("evidence-preserving reducer", () => {
	it("registers without an extension-specific credential", () => {
		const pi = new FakePi();
		expect(() => createEvidencePreservingReducerExtension()(pi.asExtensionApi())).not.toThrow();
		expect(pi.handlers.get("tool_result")).toHaveLength(1);
	});

	it("keeps the SoL-Pi identifiers that are written to disk", async () => {
		const root = await storeRoot();
		const signal = "ERROR test target failed";
		const body = `${signal}\n${"diagnostic output\n".repeat(400)}`;
		const { context, manager, pi } = load(
			root,
			modelComplete(body, (input) => ({
				schema: REDUCER_RECEIPT_SCHEMA,
				source_sha256: sourceHash(input),
				status: "failure",
				uncertain: false,
				evidence: [{ kind: "failure", quote: signal }],
			})),
		);

		const result = (await pi.emit("tool_result", bashEvent(body), context)) as {
			content: { type: string; text: string }[];
			details: Record<string, unknown>;
		};

		expect(result.content[0]?.text ?? "").toMatch(/^sol_pi_evidence_receipt_v1\n/u);
		expect(REDUCER_RECEIPT_SCHEMA).toBe("sol-pi-evidence-receipt/1");
		expect(Object.keys(result.details)).toContain("evidencePreservingReducer");
		expect(manager.entries.map((entry) => entry.type === "custom" && entry.customType)).toContain(
			"sol-pi-evidence-preserving-reducer-v1",
		);
		expect(
			manager.customEntryData().every((entry) => entry.schema === "sol-pi-evidence-preserving-reducer/1"),
		).toBe(true);
	});

	it("keeps the diagnostic command trigger generic", () => {
		expect(DIAGNOSTIC_COMMAND.test("pytest -q")).toBe(true);
		expect(DIAGNOSTIC_COMMAND.test("lake build")).toBe(true);
		expect(DIAGNOSTIC_COMMAND.test("cargo test --all")).toBe(true);
		expect(DIAGNOSTIC_COMMAND.test("rg test src")).toBe(false);
	});

	it("loads a configured reducer provider/model route", async () => {
		const root = await storeRoot();
		const config = loadReducerConfig(root, {
			reducerProvider: "test-provider",
			reducerModel: "test-reducer-model",
		});

		expect(config.reducerProvider).toBe("test-provider");
		expect(config.reducerModel).toBe("test-reducer-model");
	});

	it("uses the Luna reducer model and accepts only verified exact quotes", async () => {
		vi.useFakeTimers();
		const root = await storeRoot();
		const fatal = "E   AssertionError: expected 4 but received 5";
		const body = ["pytest session starts", fatal, "FAILED tests/test_math.py::test_addition", ".".repeat(6000)].join(
			"\n",
		);
		let call: CapturedCall | undefined;
		const notify = vi.fn();
		const setStatus = vi.fn();
		const { context, manager, pi } = load(
			root,
			modelComplete(
				body,
				(input) => ({
					schema: REDUCER_RECEIPT_SCHEMA,
					source_sha256: sourceHash(input),
					status: "failure",
					uncertain: false,
					evidence: [
						{ kind: "failure", quote: fatal },
						{ kind: "target", quote: "FAILED tests/test_math.py::test_addition" },
					],
				}),
				"stop",
				(value) => {
					call = value;
				},
			),
			ACTIVE_MODEL,
			{ mode: "tui", hasUI: true, ui: { notify, setStatus } as never },
		);

		const result = (await pi.emit("tool_result", bashEvent(body), context)) as {
			content: { type: string; text: string }[];
		};

		expect(call?.model).toBe(REDUCER_MODEL);
		expect(call?.context.systemPrompt).toContain("test/build evidence extractor");
		expect(contextInput(call!.context)).toContain("<untrusted_log>");
		expect(call?.options).toMatchObject({ cacheRetention: "none", maxTokens: 2_048, timeoutMs: 90_000 });
		expect(call?.options.signal).toBeInstanceOf(AbortSignal);
		const receipt = result.content[0]?.text ?? "";
		expect(receipt).toMatch(/status=failure/u);
		expect(receipt).toMatch(/line=2/u);
		expect(receipt).toMatch(/reducer_provider=openai-codex/u);
		expect(receipt).toContain(`reducer_model=${REDUCER_MODEL.id}`);
		expect(receipt).toMatch(/authority=the main agent retains diagnosis/u);
		expect(Buffer.byteLength(receipt)).toBeLessThan(Buffer.byteLength(body));

		const events = manager.customEntryData();
		const candidate = events.find((entry) => entry.kind === "candidate");
		expect(candidate).toBeTruthy();
		const sourcePath = String(candidate?.sourcePath);
		const localSourcePath = relative(join(runtimeRoot(context), "evidence-preserving-reducer"), sourcePath);
		expect(localSourcePath.length > 0 && !localSourcePath.startsWith("..") && !isAbsolute(localSourcePath)).toBe(true);
		expect(await readFile(sourcePath, "utf8")).toBe(body);
		expect((await stat(sourcePath)).mode & 0o777).toBe(0o600);
		expect(events.filter((entry) => entry.kind === "applied")).toHaveLength(1);
		expect(notify).toHaveBeenCalledTimes(1);
		expect(notify.mock.calls[0]?.[0]).toMatch(
			/^⚡ SoL-Pi · Evidence Reducer\nMoney saved · .+ removed from future prompts$/u,
		);
	});

	it("uses Pi-resolved authentication on a fork-shaped model registry", async () => {
		const root = await storeRoot();
		const config = loadReducerConfig(join(root, "session-runtime"));
		const body = `ERROR fork compatibility\n${"diagnostic\n".repeat(400)}`;
		const archive = await archiveBody(config.storeRoot, body);
		let call: CapturedCall | undefined;
		const completion = modelComplete(
			body,
			(input) => ({
				schema: REDUCER_RECEIPT_SCHEMA,
				source_sha256: sourceHash(input),
				status: "failure",
				uncertain: false,
				evidence: [{ kind: "failure", quote: "ERROR fork compatibility" }],
			}),
			"stop",
			(value) => {
				call = value;
			},
		) as CompatComplete;
		let authModel: Model<string> | undefined;
		const context = fakeContext(new FakeSessionManager([], "fork-session", root), {
			model: ACTIVE_MODEL,
			modelRegistry: {
				find: (provider: string, modelId: string) =>
					provider === REDUCER_MODEL.provider && modelId === REDUCER_MODEL.id ? REDUCER_MODEL : undefined,
				getApiKeyAndHeaders: async (model: Model<string>) => {
					authModel = model;
					return {
					ok: true,
					apiKey: "fork-test-key",
					headers: { "x-test-header": "fork" },
					env: { TEST_REGION: "test" },
					baseUrl: "https://fork.example.invalid/v1",
					};
				},
			} as unknown as ExtensionContext["modelRegistry"],
		});

		const result = await callReducer(config, "pytest -q", true, archive, body, context, completion);

		expect(result.ok).toBe(true);
		expect(authModel).toBe(REDUCER_MODEL);
		expect(call?.model.baseUrl).toBe("https://fork.example.invalid/v1");
		expect(call?.options).toMatchObject({
			apiKey: "fork-test-key",
			headers: { "x-test-header": "fork" },
			env: { TEST_REGION: "test" },
		});
	});

	it.each([false, true])(
		"reduces fused command output while preserving the mutation confirmation (failed=%s)",
		async (failed) => {
			const root = await storeRoot();
			const signal = failed ? "ERROR test target failed" : "PASS test target completed";
			const body = `${signal}\n${"diagnostic output\n".repeat(400)}`;
			const { context, manager, pi } = load(
				root,
				modelComplete(body, (input) => ({
					schema: REDUCER_RECEIPT_SCHEMA,
					source_sha256: sourceHash(input),
					status: failed ? "failure" : "success",
					uncertain: false,
					evidence: [{ kind: failed ? "failure" : "summary", quote: signal }],
				})),
			);

			const result = (await pi.emit("tool_result", fusedEvent(body, failed), context)) as {
				content: Array<{ type: string; text?: string }>;
				details: Record<string, unknown>;
				isError: boolean;
			};

			const projected = result.content.map((content) => content.text ?? "").join("\n");
			expect(projected).toMatch(/Successfully wrote 12 bytes to target\.ts/u);
			expect(projected).toMatch(failed ? /\[then_run:failed\]/u : /\[then_run:succeeded\]/u);
			expect(projected).toMatch(/sol_pi_evidence_receipt_v1/u);
			expect(projected).not.toContain(body);
			expect(projected).toContain("source_tail_utf8=");
			expect(result.isError).toBe(failed);
			expect(result.details.patch).toBe("test patch");
			const candidate = manager.customEntryData().find((entry) => entry.kind === "candidate");
			expect(await readFile(String(candidate?.sourcePath), "utf8")).toBe(body);
		},
	);

	it.each(["invented", "model-error"] as const)("fails open on %s", async (mode) => {
		const root = await storeRoot();
		const body = `ERROR real failure\n${"x".repeat(5000)}`;
		const { context, manager, pi } = load(
			root,
			modelComplete(
				body,
				(input) => ({
					schema: REDUCER_RECEIPT_SCHEMA,
					source_sha256: sourceHash(input),
					status: "failure",
					uncertain: false,
					evidence: [{ kind: "failure", quote: "ERROR invented failure" }],
				}),
				mode === "model-error" ? "error" : "stop",
			),
		);

		const result = await pi.emit("tool_result", bashEvent(body), context);

		expect(result).toBeUndefined();
		const fallbacks = manager.customEntryData().filter((entry) => entry.kind === "fallback");
		expect(
			fallbacks.some((entry) =>
				mode === "model-error"
					? entry.reason === "model-response-error" && entry.stopReason === "error"
					: entry.reason === "unverifiable-quote",
			),
		).toBe(true);
	});

	it("fails open when Pi cannot complete the nested model call", async () => {
		const root = await storeRoot();
		const body = `ERROR real failure\n${"x".repeat(5000)}`;
		const { context, manager, pi } = load(root, async () => {
			throw new Error("authentication is not configured");
		});

		expect(await pi.emit("tool_result", bashEvent(body), context)).toBeUndefined();
		expect(manager.customEntryData()).toContainEqual(
			expect.objectContaining({ kind: "fallback", reason: "model-call-exception" }),
		);
	});

	it("fails open when the configured reducer model is unavailable", async () => {
		const root = await storeRoot();
		const body = `ERROR real failure\n${"x".repeat(5000)}`;
		let calls = 0;
		const { context, manager, pi } = load(
			root,
			async () => {
				calls++;
				throw new Error("unexpected model call");
			},
			ACTIVE_MODEL,
			{
				modelRegistry: {
					find: () => undefined,
					complete: async () => {
						calls++;
						throw new Error("unexpected model call");
					},
				} as unknown as ExtensionContext["modelRegistry"],
			},
		);

		expect(await pi.emit("tool_result", bashEvent(body), context)).toBeUndefined();
		expect(calls).toBe(0);
		expect(manager.customEntryData()).toContainEqual(
			expect.objectContaining({ kind: "fallback", reason: "reducer-model-unavailable" }),
		);
	});

	it("fails open when Pi has no persistent session directory", async () => {
		const manager = new FakeSessionManager([], "ephemeral-session", "");
		const pi = new FakePi(manager);
		createEvidencePreservingReducerExtension()(pi.asExtensionApi());
		let calls = 0;
		const context = fakeContext(manager, {
			model: ACTIVE_MODEL,
			modelRegistry: {
				complete: async () => {
					calls++;
					throw new Error("unexpected model call");
				},
			} as unknown as ExtensionContext["modelRegistry"],
		});
		const body = `ERROR no session storage\n${"x".repeat(5000)}`;

		expect(await pi.emit("tool_result", bashEvent(body), context)).toBeUndefined();
		expect(calls).toBe(0);
	});

	it("reads only Pi output files in the system temporary directory", async () => {
		const root = await storeRoot();
		const fullBody = `ERROR full output\n${"full diagnostic\n".repeat(400)}`;
		const outputPath = join(tmpdir(), `pi-bash-${randomUUID()}.log`);
		await writeFile(outputPath, fullBody, { mode: 0o600 });
		cleanupPaths.push(outputPath);
		let input = "";
		const { context, manager, pi } = load(
			root,
			modelComplete(fullBody, (value) => {
				input = value;
				return {
					schema: REDUCER_RECEIPT_SCHEMA,
					source_sha256: sourceHash(value),
					status: "failure",
					uncertain: false,
					evidence: [{ kind: "failure", quote: "ERROR full output" }],
				};
			}),
		);

		await pi.emit(
			"tool_result",
			bashEvent("ERROR truncated", { details: { fullOutputPath: outputPath } }),
			context,
		);
		expect(input).toContain("1\tERROR full output\n2\tfull diagnostic");
		const candidate = manager.customEntryData().find((entry) => entry.kind === "candidate");
		expect(await readFile(String(candidate?.sourcePath), "utf8")).toBe(fullBody);

		const outsidePath = join(root, `pi-bash-${randomUUID()}.log`);
		await writeFile(outsidePath, `ERROR outside file\n${"outside\n".repeat(600)}`);
		const inlineBody = `ERROR inline output\n${"inline diagnostic\n".repeat(400)}`;
		input = "";
		await pi.emit(
			"tool_result",
			bashEvent(inlineBody, { toolCallId: "call-2", details: { fullOutputPath: outsidePath } }),
			context,
		);
		expect(input).toContain("1\tERROR inline output\n2\tinline diagnostic");
		expect(input).not.toContain("ERROR outside file");
	});

	it("does not delegate small or non-diagnostic output", async () => {
		const root = await storeRoot();
		let calls = 0;
		const { context, manager, pi } = load(root, async () => {
			calls++;
			throw new Error("unexpected model call");
		});

		expect(await pi.emit("tool_result", bashEvent("ERROR short"), context)).toBeUndefined();
		expect(
			await pi.emit(
				"tool_result",
				bashEvent("x".repeat(5000), { input: { command: "rg symbol src" } }),
				context,
			),
		).toBeUndefined();
		expect(calls).toBe(0);
	});

	it("keeps exact ending bytes when the model omits aggregate results", async () => {
		const root = await storeRoot();
		const body = `ERROR arithmetic failed\n${"diagnostic λ🙂\n".repeat(600)}2 failed, 598 passed in 4s\n`;
		const { context, pi } = load(root, modelComplete(body, (input) => ({
			schema: REDUCER_RECEIPT_SCHEMA, source_sha256: sourceHash(input), status: "failure", uncertain: false,
			evidence: [{ kind: "failure", quote: "ERROR arithmetic failed" }],
		})));
		const result = await pi.emit("tool_result", bashEvent(body), context) as { content: { text: string }[] };
		const receipt = result.content[0]!.text;
		const tailLine = receipt.split("\n").find((line) => line.startsWith("source_tail_utf8="))!;
		const tail = JSON.parse(tailLine.slice("source_tail_utf8=".length)) as string;
		expect(body.endsWith(tail)).toBe(true);
		expect(Buffer.byteLength(tail)).toBeLessThanOrEqual(1536);
		expect(tail).toContain("2 failed, 598 passed in 4s");
		expect(receipt).toContain("completeness and diagnosis are not verified");
	});

	it("copies selected original lines without asking the model to retype them", async () => {
		const root = await storeRoot();
		const body = "test starts\n  ERROR: exact spacing λ\n\n2 tests failed\n";
		const archive = await archiveBody(root, body);
		const receipt = {
			schema: REDUCER_RECEIPT_SCHEMA, source_sha256: archive.hash, status: "failure", uncertain: false,
			evidence: [{ kind: "failure", line: 2 }, { kind: "summary", line: 4 }],
		};
		const result = validateReceipt(JSON.stringify(receipt), archive, body, true);
		expect(result.ok).toBe(true);
		if (result.ok) expect(result.value.evidence.map(({ line, quote }) => ({ line, quote }))).toEqual([
			{ line: 2, quote: "  ERROR: exact spacing λ" }, { line: 4, quote: "2 tests failed" },
		]);
		for (const line of [0, 1.5, 999]) {
			expect(validateReceipt(JSON.stringify({ ...receipt, evidence: [{ kind: "failure", line }] }), archive, body, true))
				.toEqual({ ok: false, reason: "invalid-evidence-line" });
		}
		expect(validateReceipt(JSON.stringify({ ...receipt, evidence: [{ kind: "failure", line: 2, quote: "ERROR: exact spacing λ" }] }), archive, body, true))
			.toEqual({ ok: false, reason: "unverifiable-quote" });
		expect(validateReceipt(JSON.stringify({ ...receipt, evidence: [{ kind: "failure", line: 1 }] }), archive, body, true))
			.toEqual({ ok: false, reason: "missing-failure-evidence" });
	});

	it.each(["uncertain", "empty"] as const)("keeps the original output for %s evidence", async (mode) => {
		const root = await storeRoot();
		const body = `build complete\n${"details\n".repeat(700)}`;
		const { context, manager, pi } = load(root, modelComplete(body, (input) => ({
			schema: REDUCER_RECEIPT_SCHEMA, source_sha256: sourceHash(input), status: "success",
			uncertain: mode === "uncertain", evidence: mode === "empty" ? [] : [{ kind: "summary", quote: "build complete" }],
		})));
		expect(await pi.emit("tool_result", bashEvent(body, { isError: false }), context)).toBeUndefined();
		expect(manager.customEntryData()).toContainEqual(expect.objectContaining({
			kind: "fallback", reason: `${mode}-evidence`,
		}));
	});

	it.each(["unwritable", "corrupt", "symlink"] as const)("keeps original output when its archive is %s", async (mode) => {
		const root = await storeRoot();
		const body = `ERROR preserved original\n${"diagnostic\n".repeat(600)}`;
		const complete = vi.fn(async () => { throw new Error("unexpected model call"); });
		const { context, manager, pi } = load(root, complete);
		const config = loadReducerConfig(runtimeRoot(context));
		const archive = await archiveBody(config.storeRoot, body);
		if (mode === "unwritable") {
			await rm(config.storeRoot, { recursive: true });
			await writeFile(config.storeRoot, "a file blocks the archive directory");
		} else if (mode === "corrupt") {
			await writeFile(archive.path, "different bytes");
		} else {
			const target = join(root, "same_log.txt");
			await writeFile(target, body);
			await rm(archive.path);
			await symlink(target, archive.path);
		}
		expect(await pi.emit("tool_result", bashEvent(body), context)).toBeUndefined();
		expect(complete).not.toHaveBeenCalled();
		expect(manager.customEntryData()).toContainEqual(expect.objectContaining({
			kind: "fallback", reason: "archive-unavailable",
		}));
	});

	it("journals the provider failure category without its arbitrary error body", async () => {
		const root = await storeRoot();
		const body = `ERROR preserved original\n${"diagnostic\n".repeat(600)}`;
		const base = modelComplete(body, (input) => ({
			schema: REDUCER_RECEIPT_SCHEMA, source_sha256: sourceHash(input), status: "failure", uncertain: false,
			evidence: [{ kind: "failure", quote: "ERROR preserved original" }],
		}), "error");
		const { context, manager, pi } = load(root, async (...args) => ({
			...await base(...args), errorMessage: "synthetic_private_error_canary",
		}));
		expect(await pi.emit("tool_result", bashEvent(body), context)).toBeUndefined();
		const entries = manager.customEntryData();
		expect(entries).toContainEqual(expect.objectContaining({ reason: "model-response-error", stopReason: "error" }));
		expect(JSON.stringify(entries)).not.toContain("synthetic_private_error_canary");
		expect(entries.every((entry) => !("errorMessage" in entry))).toBe(true);
	});
});
