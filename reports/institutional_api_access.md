# Institutional API access

2026-09-26. The saved `LLM Manager.html` supplies the missing general-information and API quick-start text. It is sufficient for connection setup. The separate catalogue page was not needed to discover accessible model IDs because the authenticated model-list call succeeded.

## Verified access

- Base URL: `https://llm.iiia.es/v1`.
- Authentication: Bearer token from external `IIIA_API_KEY`.
- API: OpenAI-compatible `GET /models` and `POST /chat/completions`, as documented in the supplied page.
- Actual authenticated `GET /v1/models`: HTTP 200 at 2026-09-26 13:27:30 UTC. TLS validation remained enabled; redirects were disabled to avoid forwarding the credential elsewhere.
- The Bash assignment now has no whitespace before `=` and is compatible with the existing loader. The agent did not edit `.bashrc`, source it, print the key or save its value.
- Selected-model generation was verified after explicit goal resumption: the first bounded request returned `deepseek-ai/DeepSeek-V4-Flash-0731` with finish reason `stop`. This does not establish generation availability for other listed models.

## Returned model IDs

- `deepseek-ai/DeepSeek-V4-Flash`
- `deepseek-ai/DeepSeek-V4-Flash-0731`
- `Qwen/Qwen3.8-27B`
- `Qwen/Qwen3.8-27B-FP8`
- `XiaomiMiMo/MiMo-V2.6-Flash`
- `XiaomiMiMo/MiMo-V2.6-Flash-RL`
- `default/llm`

No Gemini model is listed for this key. The `default/llm` alias should not identify the research target. An explicit returned ID is required, and a dated name still does not establish an immutable weights revision. Generic `created`/`owned_by` compatibility metadata was not treated as the actual model revision or hosting provenance.

## Cost and selected target

The supplied general-information page (updated 5 August 2026) says current models are hosted on CSIC scientific-computing hardware and are free to users. Commercial models require separate project billing and enablement; that has not been authorized. No numeric quota is given in the supplied section, so quota remains unresolved rather than assumed unlimited.

The user explicitly selected `deepseek-ai/DeepSeek-V4-Flash-0731`, superseding Gemini, then later resumed the goal. The bounded institutional development pilot is complete under the unchanged 256-token cap; no model-selection comparison was run. The existing environment uses the documented API without a new SDK. The earlier offline validation is recorded in `reports/deepseek_adapter_validation.md`; live requests, ledger, responses and judgments use the separate `results/goal2/application_deepseek/` directory. All previous Gemini evidence is preserved separately. The finalized development outcomes are in `summary.json` and `diagnostics.json`; they do not constitute final held-out evaluation.

Listing evidence: `results/goal2/institutional_model_access.json` and `results/goal2/institutional_access_check.json`. Live evidence: `results/goal2/application_deepseek/target_responses.jsonl`. Current execution and usage-reset status are maintained in `STATUS.md`.

## Historical stop and completed resumption

The earlier model-selection comparison ended at the user's request and consisted only of documentation review. The adapter then passed offline checks with a zero-call preview while dispatch was stopped. Those statements describe the prior handoff, not the current execution state. The user subsequently resumed the selected DeepSeek experiment. The resumed run has now completed. Shared config preserves the prior Gemini reference and keeps the immutable revision unresolved; `STATUS.md` records the remaining research work.

## Generation verified after explicit study resumption

On 2026-09-26 the user explicitly resumed the goal. The first bounded development response, saved at 14:13:54 UTC, returned the exact selected DeepSeek model label with `finish_reason=stop` and ID `chatcmpl-8904ab8ced20ea70`. Raw usage was 186 prompt tokens and 189 completion tokens. This establishes selected-model generation access and acceptance of the submitted request body. It does not independently prove that individual decoding parameters were honored, and `immutable_model_revision` remains null. This first request is bring-up history. The completed pilot evidence below supersedes its one-request progress snapshot.

## Completed bounded pilot

The finalized saved records contain 120 requests with no API errors or retries: 43 finished normally and 77 truncated at the fixed cap, including 64 with no visible answer. All 120 responses have completed rubric judgments and returned the exact selected model label. `immutable_model_revision` remains null; accepted requests do not independently prove individual parameter honoring. All responses contain reasoning chunks while all usage records report zero reasoning tokens, so the reasoning-token breakdown cannot be taken literally.

Reported usage totals are 29,168 input and 28,006 completion tokens. USD 0 is the recorded estimate for the institutional-included route, not a billing receipt or proof of unlimited quota. Clean completion is 9/20 distinct tasks (45/100 repeated logical cases per clean condition). Unchecked attacked cases have 24/100 task completion and 11/100 attack success. Checking rejects all 100 attacked chunks, yielding zero attack success and zero completion, with no clean rejections. These completed development rates have strong truncation and sample-size limitations.

Authoritative evidence: `results/goal2/application_deepseek/summary.json` (regenerated 2026-09-26 14:47:45 UTC), `diagnostics.json`, raw responses and saved judgments. Goal 2's required detector and four-condition pilot are complete; optional Prompt Guard remains unavailable. The default application command rebuilds the report from saved evidence offline. No additional API request is needed for this pilot.
