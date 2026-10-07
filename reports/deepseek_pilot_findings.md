# DeepSeek development pilot findings

2026-09-26. The required Goal 2 detector and four-condition development pilot are complete under the recorded settings. This is a bounded development result, not final evaluation or a model-selection comparison. The institutional API worked: 120 distinct requests returned the selected `deepseek-ai/DeepSeek-V4-Flash-0731` label, with no HTTP/transport errors, retries or unresolved dispatches. All 120 responses have exact response-ID-bound Codex rubric judgments.

## Application outcomes

The pilot contains 100 clean/attacked pairs from 20 source tasks, with five attack variants per source. Clean answers are reused across those variants for quality only, so the 100 clean cases represent 20 distinct model requests. Four conditions produce 400 logical cases from 120 actual requests.

| Condition | Logical cases | Guard rejected | Reference-based task completion | Attack success, all attacked cases |
|---|---:|---:|---:|---:|
| Clean, no guard | 100 | 0 | 45/100 (9/20 distinct tasks) | Not applicable |
| Clean, guard | 100 | 0 | 45/100 (9/20 distinct tasks) | Not applicable |
| Attacked, no guard | 100 | 0 | 24/100 | 11/100 |
| Attacked, guard | 100 | 100 | 0/100 | 0/100 |

The guard removed the sole evidence chunk from every attacked task and returned the explicit insufficient-evidence response. It prevented all 11 attacks that succeeded without the guard, while also preventing legitimate completion of those attacked tasks. This shows the protection/utility trade-off of whole-chunk rejection in this fixed development setting. It does not establish utility-preserving sanitization or general robustness. No clean pilot source was rejected; the authoritative larger floating-detector scoring FPR remains 18/1,369, above the 1% requirement.

Ten strict attack successes were full injected safety sentences (10/20 for that payload); one was the requested Spanish translation (1/20). The other three fixed payload rubrics had zero successes in 20 cases each. These are dependent variants of 20 sources, not 100 independent samples. Two truncated answers began the safety sentence without completing it. Their primary attack judgments remain false under the fixed rubric, but the partial compliance is preserved in the notes and must not be described as ignoring the injection.

## Output cap and reference limitations

There were 43 normal completions and 77 length-truncated responses: 7/20 clean and 70/100 attacked. Of the truncations, 64 had no visible answer and 13 contained partial visible text. Every non-ok response is a task non-completion under the fixed policy. Fully received length-ended streams with empty visible answers did not produce the attack target; reasoning text was not scored as the application answer. The 11% unguarded attack-success rate therefore describes this 256-token configuration and cannot be generalized to a larger output allowance.

All 100 prepared pairs match the pinned source questions, references, contexts and declared insertions. That does not establish semantic correctness of every upstream reference. Wise and Stripe references attribute amounts to companies absent from the email; Corp E has a payer/charger wording conflict. All original references were retained, so these task scores measure agreement with the fixed benchmark criterion and do not settle real-world answer correctness for those cases. See `reports/application_reference_audit.md` and individual judgment notes. No alternate scoring rule was substituted after seeing outcomes.

## Protocol, accounting and timing

- Endpoint: `https://llm.iiia.es/v1/chat/completions`; exact requested and returned model label: `deepseek-ai/DeepSeek-V4-Flash-0731`.
- Fixed original BIPIA role placement and text, temperature 0, requested reasoning effort low, maximum 256 completion tokens, streaming, no tools or grounding. Acceptance of the request does not independently prove every parameter was honored.
- All 120 streams included reasoning text, while the provider reported `reasoning_tokens=0`. Raw fields are retained; no independent reasoning-token count is claimed. Completion tokens are charged once, without adding reasoning again.
- Actual reported usage: 29,168 input and 28,006 completion tokens. Original caps were 120 attempts, 153,296 input and 30,720 output tokens. Estimated charge is USD0 under documented institution-hosted included access; no commercial fallback was used. This is not a billing receipt or a claim of unlimited quota.
- All returned model labels match; no immutable weights revision or system fingerprint was supplied. This bounds exact reproduction of the hosted model.
- Across the 43 status-ok responses, mean first-visible/full-response latency was 1.4736/1.5851 seconds. Across all 120 requests, full-response latency averaged 1.8513 seconds. These include provider/network time; they are not end-to-end guard or FPGA latency. Empty visible answers have no first-visible latency. Cached quality reuse has null execution latency.

## Evidence and reproduction

Authoritative artifacts in `results/goal2/application_deepseek/`: `request_plan.json`, `case_plan.jsonl`, `api_events.jsonl`, `target_responses.jsonl`, `manual_judgments.jsonl`, `outcomes.jsonl`, `summary.json`, and `diagnostics.json`. The generated protocol/outcome report is `reports/application_pilot_deepseek.md`. Prior Gemini results remain separate and support no head-to-head inference here.

Regenerate the report offline, with no credential or network call:

```powershell
& 'C:\research\GATE\.venv\Scripts\python.exe' '\\wsl.localhost\Ubuntu\home\jp19126\Projects\SENTRY\scripts\application_pilot.py'
```

No further API request is needed to reproduce this completed pilot. Final calibration/test, NotInject, hardware costs, A/B/C comparison, FPGA measurements and final held-out application evaluation remain unexecuted. The optional gated Prompt Guard reference remains unavailable. No full research-goal completion is claimed.
