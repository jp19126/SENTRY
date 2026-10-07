# DeepSeek adapter: offline validation

2026-09-26. The separately recorded target is `deepseek-ai/DeepSeek-V4-Flash-0731` at `https://llm.iiia.es/v1`. This handoff performed **zero API requests, zero credential reads and zero model benchmarks**. Generation remains stopped: `application_pilot.authorized=false`, `benchmark_dispatch_stopped_by_user=true`, and `results/goal2/pause.request` still exists.

Changed source: `scripts/application_pilot.py`. The institutional request uses the exact existing BIPIA system/user text, Chat Completions streaming, temperature 0, `max_tokens=256`, `reasoning_effort=low`, and no tools. Provider, approved endpoint and requested model are recorded separately. The existing Gemini format remains readable, including its older request-plan schema.

## Actual checks

Syntax was checked with Python's `compile(source, path, "exec")`. Narrow Python fixtures were sent through standard input using:

```text
C:\research\GATE\.venv\Scripts\python.exe -
```

All checks passed:

- The prepared inputs still produce 120 distinct requests, 400 logical cases and 100 guard-rejected attacked cases. The exact system/user text and token reservations match the preserved Gemini request template.
- The documented institutional USD0 route validates its selected endpoint/model and billing evidence. Paid use with a USD0 allowance is rejected.
- HTTP, unapproved hosts/paths, embedded URL credentials and authentication redirects are rejected. Institutional output/report paths cannot overwrite the historical Gemini paths.
- Local SSE fixtures separate reasoning from visible text. Visible-answer TTFT starts with visible content; the terminal `[DONE]` marker stops further reads. These fixture clock values are not endpoint timing measurements.
- Raw OpenAI `model`, `id`, `system_fingerprint`, `usage` and finish reason survive parsing. A fixture with 11 completion tokens, including 7 reasoning tokens, accounts for 11 output tokens, not 18. Gemini's separate candidate/thought counts remain supported.
- Mismatched or changed model labels stop processing. Visible or reasoning-only partial responses missing model/id also stop further dispatch. No-response identity consistency remains null; a matching complete response plus a partial response missing identity reports false, while a matching identified response alone reports true. This final independent-review correction passed a narrow in-memory check. Immutable served revision remains unresolved.
- A completed result event is recovered if interruption occurs before the response-file append. A later result supersedes an older failure; conflicting records for the same attempt are rejected.
- The full `main --execute --max-new-requests 1` branch was exercised **only with fail-fast credential/transport stubs and in-memory output sinks**. The existing user stop returned before either stub: zero dispatches, zero results and no fixture artifact writes.

The initial offline preview command, run from the repository, was:

```text
C:\research\GATE\.venv\Scripts\python.exe scripts/application_pilot.py
```

It exited successfully and wrote only the configured DeepSeek preview artifacts under `results/goal2/application_deepseek/` and `reports/application_pilot_deepseek.md`. Recorded outcome: 153,296 conservative input tokens plus 30,720 output/reasoning tokens reserved, estimated USD0, zero attempts, zero new requests, four-condition execution incomplete, and dispatch stopped by the user. No response or timing from Gemini was used as a DeepSeek result.

Before and after the preview, all ten files under `results/goal2/application/` and the historical `reports/application_pilot.md` had exactly unchanged file sizes and modification times. No hashes or integrity manifest were created. The adapter task did not edit shared configuration or stop controls. At the final handoff, the root agent updated readiness/status metadata in the canonical and existing native configuration; all stop controls remained unchanged.

## Remaining uncertainty

Local fixtures validate the adapter, not the live institution-hosted implementation. This work does not establish acceptance of `reasoning_effort`, `max_tokens` or `stream_options`, actual streaming field behavior, quotas, model output quality or latency. Returned model labels and fingerprints will not establish an immutable weight revision. Any unsupported parameter, changed/missing output identity, quota failure or permanent request/authentication failure stops the run without a commercial or model fallback.

Actual generation requires explicit user resumption and the configured authorization path. No such resumption occurred in this handoff.

## Final handoff verification

The independent evaluator reviewed the final source and found no remaining concrete defect within the requested offline scope. It directly compared the saved DeepSeek and historical Gemini plans: 120 identical request IDs, exact system/user text, and identical mappings for all 400 logical cases. Following the final status update, root ran a second offline preview using the absolute canonical script path with the existing native Python. It exited successfully with zero API attempts, zero saved target responses and all stop controls intact. All 11 protected Gemini files again had unchanged sizes and modification times. Canonical and native shared configs match. This final report regeneration adds no model measurement or endpoint validation.
