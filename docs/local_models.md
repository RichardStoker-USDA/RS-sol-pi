# Local-model profile

This fork keeps the upstream mechanisms and adds explicit controls for a local model used for coding tasks. It is tested against Pi 0.85.1. Pi itself is unmodified.

SoL-Pi improves how an agent uses tools and context. It does not retrain the model, add knowledge, or guarantee frontier-model reasoning. NVIDIA's [published results](https://nvlabs.github.io/SoL-Pi/#results) include both efficiency gains and some lower task scores. Those results used other models and do not measure this profile.

Use this effective `sol-pi.json` for single-agent work:

```json
{
  "version": 1,
  "actionFusion": true,
  "observationPack": true,
  "observationPackHeadBytes": 2048,
  "observationPackTailBytes": 1536,
  "evidencePreservingReducer": false,
  "onlineContextCompact": true,
  "cacheWriteReadRatio": null,
  "onlineContextCompactKeepRecentTokens": 65536,
  "onlineContextCompactReserveTokens": 65536
}
```

Match Pi's `compaction.keepRecentTokens` and `compaction.reserveTokens` to 65536 for this profile. These values are intended for a 512K model window, not a small-context model. Native overflow recovery remains enabled. The plan tool can trigger compaction at a completed-step boundary when context pressure warrants it; token price alone cannot trigger it. A `null` ratio is different from zero: zero makes price-driven compaction inexpensive and therefore frequent.

The larger excerpts match the blog's selected Observation Pack V2 budgets. Original results still receive two full sends. Successful pure-text results above 10 KiB then become a stable placeholder with exact paged recall. Failed tool results and image-bearing results remain intact. Saved session messages retain the original text. Stored observations survive session resume and native compaction; they are local files, not the server's KV cache.

The Evidence-Preserving Reducer is disabled because it invokes another model and replaces fresh diagnostics with a receipt. Hash and quote checks prove that cited quotes exist, but cannot prove the reducer retained every relevant detail. Enabling it later needs explicit permission to delegate and a comparison on representative logs. Set an explicit local provider/model route before enabling; the upstream default route is a remote provider. No second GPU is needed by Action Fusion, Observation Pack, or Online Context Compact.

### Optional local diagnostic reducer

After the user authorizes this narrow model call, choose an existing local provider that passes a representative diagnostic check. Configure the provider in Pi and set these fields in the effective SoL-Pi config:

```json
{
  "evidencePreservingReducer": true,
  "evidencePreservingReducerProvider": "local-provider",
  "evidencePreservingReducerModel": "local-model"
}
```

Keep the rest of the effective configuration. The names in the example refer to the provider and model
registered in Pi. The reducer requests a 2,048-token output limit without a reasoning budget. The main agent
keeps its own settings.

Reduction applies only to eligible build/test command results of at least 4,096 bytes and at most 600,000 characters. It verifies the source hash, observed success/failure, and exact quotes. The fork also includes up to 1,536 exact UTF-8 bytes from the log's ending independently of the model's selection. This preserves common final counts even when the model omits them. Uncertain, empty, invalid, unavailable, or non-shrinking reductions leave the original tool output in place. Archive errors also retain the original output; arbitrary provider error bodies are excluded from the journal. Original logs remain in private session storage for bounded readback.

The reducer selects numbered original lines; code copies those lines into the quoted receipt. This avoids asking a model to retype formatting or join summary lines. Invalid line selections are rejected, and an optional returned quote must match the selected line exactly. Legacy exact-quote responses remain supported. Whitespace checks are never relaxed to accept a summary.

This can reduce repeated prompt volume. It adds a model call and does not establish greater reasoning accuracy or complete evidence coverage. Treat logs, including their quotes, as untrusted data. The sensitive-text heuristic is not a comprehensive secret detector. Enable only a provider allowed to receive those logs. A separate helper GPU is useful only when its model passes the evidence checks at an acceptable latency.

## Verify the profile

Run `npm ci --ignore-scripts` and `npm run check` for the source checks and package file list.
Validate your effective settings with `node scripts/check-sol-pi-config.mjs --config /path/to/sol-pi.json`.
Use a representative diagnostic log and check the archived output and exact quotes before enabling the
reducer for routine work.
