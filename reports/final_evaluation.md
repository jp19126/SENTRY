# Goal 7 final evaluation preparation

2026-10-08. The quality and held-out application entry points are implemented.
No final-calibration/test/NotInject/NewsQA content was opened for this work, and
no model, held-out score, application generation or board benchmark ran. This
report records runnable preparation and the actual prerequisite checks, not
final quality or completion of Goal 7.

## Actual gates and development freeze

`scripts/evaluate_frozen_finalists.py` previews by default. Its actual shared
configuration check reports: quality readiness false, unresolved approved score
identities, and no real `results/goal5/development_freeze.json`. The historical NewsQA source remains absent; the active public TableQA source is
now available, with selection and scoring still gated on the actual freeze. `--check-prerequisites` returns 2 on these
blockers; ordinary preview returns 0. The application check has the same quality
blockers and defaults to **zero new authorized calls**. Preview with a prospective
one-call limit additionally reports that the continuation excludes additional
target calls and that a separate Goal 7 allowance is missing. It does not call
the API, read credentials or inherit the completed development pilot's120 calls.

Both entry points, including their `--prepare` actions and the NewsQA
preparation gate, were exercised against the actual configuration with a
read-only file-open guard that would fail on a held-out source read. A network trap also rejected any unexpected access. All checks
passed with zero held-out reads, no `torch` import/model loading and no Goal 7
output directories. Python source compilation also passed. These are functional
prepare/gate checks, not fixture-based experimental results or final inference.

The coordinator must create the freeze from actual development evidence. No
placeholder freeze is generated. Shared `final_evaluation.development_freeze_path`
selects its path. Its schema is:

- `state`: `frozen_development`.
- `evaluation` and `windowing`: exact shared configuration snapshots.
- `mechanism`: the actual selected C mechanism, matching shared configuration.
- `finalists`: exactly `floating`, `U`, `B`, `C`. Each contains `precision_map`,
  `checkpoint`, `identity`, and `development_scores` (candidate folder). Floating
  has null precision and its selected floating checkpoint. U/B/C identify actual
  post-QAT checkpoints/maps; U is uniform.

The gate validates existing approved arithmetic/execution identities, selected
checkpoint and common QAT associations using the existing search helper. It
requires matching real candidate files and recomputes their development metrics
from the actual prepared-document-associated score rows. All gates run before
held-out content or model work. Quality readiness is independent of a physical
board claim; this driver does not silently add board access as a prerequisite.
Existing result/freeze/identity mismatches stop instead of relabelling evidence.

## Final quality and bounded challenge

After a real freeze and readiness approval, `--prepare` writes only an execution
plan. `--execute --phase quality` scores benign final calibration first, saves each
method's thresholds atomically, then scores test and NotInject separately. It
uses the existing document evaluator, ordered risk function and actual frozen
model/checkpoint. The backend uses the explicit shared scoring batch policy (currently configured
CPU B32, with score approvals still unresolved). B8/B32 validate the exact frozen
role's precision map, QAT checkpoint, own batching proof and approved execution
identity before model work. The proof is checked once; model loading then uses
`scoring=False` and adds the verified batch suffix once, followed by exact full
identity comparison. Floating uses its own reference proof. This avoids reusing a
uniform proof for a mixed finalist. Source evidence is in
`results/goal7/finalist_batch_binding_source_v1/` and
`results/goal7/final_driver_source_review_v1/source_review.md`. Model loading is lazy;
completed chunks are reused. A completed next chunk is recoverable if an
interruption occurs before progress commit. Default allowance is32 new documents
per invocation, in chunks of16; it never expands into training or a full search.

Scores are document maxima and rejection is strictly `score > threshold`.
Threshold fitting uses only benign final-calibration scores. Primary and nearby
FPR thresholds use the fixed order-statistic rule; test and NotInject metrics
reuse the same cached scores. Saved outputs include confusion counts, actual
source groups, source strata and single/multiple-window strata. Paired B-minus-U
and C-minus-B changes include500 paired source-group bootstrap resamples and95%
percentile intervals, conditional on frozen calibration thresholds. Windows are
never resampled as independent documents. Prepared-data counts in `reports/data.md`
remain the source of the known benign-count shortfall; this implementation does
not invent new benign test bodies.

`--phase challenge` requires complete cached test scores. It deterministically
selects20 actual held-out BIPIA source groups using seed+701 and at most three
existing attack variants per group. The selected document IDs, positions and
attack templates are saved and shared across all methods. Actual available
variant counts are retained. It reports guard outcomes and whether any variant
in a group passes the guard. This is explicitly **nonadaptive**, has no targeted
detector and makes no LLM attack-success claim. No new attacks or API calls are
created to meet the bound.

## One extension: pinned public BIPIA TableQA

The coordinator selected **one TableQA task** before method findings or extension
evaluation, replacing the unrealized NewsQA access-dependent plan. No NewsQA
selection or scores were realized. Its historical preparation functions/evidence
remain, but the active driver refuses NewsQA CLI routes and runs no second
extension. The shared configuration selection is owned by the coordinator.

The public [pinned TableQA test file](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/benchmark/table/test.jsonl)
was retrieved once at revision a004b69ec0dd446e0afd461d98cb5e96e120a5d0.
Schema inspection found 100 rows and84 distinct whitespace-normalized table
contexts, each row with string context, question and ideal. No body/answer examples were displayed, selected or scored.
The [benchmark README](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/benchmark/README.md)
identifies TableQA as a public supplied test/train task. The
[pinned table builder](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/bipia/data/table.py)
uses context as the table/form text and retains question and ideal separately.
The adapter preserves original serialized context and all associated source
rows/questions/references. Detector input remains document context only.

After the actual validated development freeze and explicit readiness,
--prepare-tableqa deterministically selects exactly20 distinct
whitespace-normalized source groups using seed+702. Exact normalized source
bodies are excluded against main train, both search subsets, final-calibration
and test records, including retained task contexts. Each group supplies one
clean document and two original held-out BIPIA payload variants, one start and
one end, using existing seed+703 sampling and upstream newline insertion.
No table cell/row is reserialized and no attack uses detector feedback.
The resulting60 documents, actual freeze, source rows, payload IDs and positions
are saved once and reused identically by floating/U/B/C.

--execute --phase extension additionally requires all main final-calibration
thresholds already frozen. It reuses frozen checkpoints, maps, thresholds,
strict score>threshold, window policy and document maximum. It performs no
refitting, training, map selection, target generation or hardware work; extension
results cannot feed development. Selection/scoring have **not** run. Differing
existing pools are preserved and rejected.

The [pinned BIPIA license](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/LICENSE)
licenses code under MIT but explicitly gives WikiTableQuestions data its own
[CC BY-SA 4.0 terms](https://github.com/ppasupat/WikiTableQuestions/blob/master/LICENSE).
Retain Microsoft/BIPIA and WikiTableQuestions attribution, source/license links
and marked start/end injection adaptations; shared adapted data remains under
CC BY-SA 4.0. These data are not described as MIT-only. Actual retrieval, primary
source/license copies, before/diff, schema and blocked stdlib previews are in
results/goal7/tableqa_extension_source_v1/. No credentials, payment or target API
is required.

## Held-out application preparation and separate generation gate

`scripts/run_frozen_application.py` uses the same deterministic20 held-out groups
and actual available variants. Each EmailQA task retains its one naturally
supplied chunk, source question/reference and payload. It reuses the pilot's
request/prompt assembly, four conditions, conservative reservations, billing and
provider identity checks, stream parser, response recovery and fixed rubrics.
U/B/C use their complete cached document scores and final-calibration thresholds.
All required guard checks must complete before generation. Identical target
requests share saved responses for quality only, across conditions/methods.

`--prepare` writes a frozen request plan and pending/resolved guard case metadata.
The immutable plan excludes incidental preparation timing. Pending guard metadata
may become ready while the source/model/request plan stays fixed. A previously
resolved guard result cannot change on resume. Saved responses and ledger events
are append-only; old answers are not overwritten or retried automatically.

A future generation run additionally requires:

- Current continuation explicitly authorizes additional target API calls.
- `--authorization PATH` identifies a separate file with `scope` equal to
  `heldout_application_goal7`, `authorized: true`,
  `previous_pilot_allowance_reused: false`, and an actual `authorization_note`.
- Its `target` exactly matches `application_pilot.target_settings(config)`, and it
  supplies positive total `max_api_calls`, `max_input_tokens`,
  `max_output_tokens_total`, plus a finite nonnegative `budget_usd`.
- `--max-new-authorized-calls` is explicitly positive. The global new allowance
  remains charged through the separate Goal 7 ledger; an invocation cap cannot
  reset it. Original pilot prices, billing basis, endpoint/model restrictions,
  credential rules and live stop checks remain in force.

No credential is read and no generation occurs under the default zero allowance.
A changed allowance, unresolved earlier dispatch, provider/identity stop, quota,
authentication or configuration error stops dispatch for review. The completed
pilot's120 attempts remain intact in their original directory.

Application outcomes retain document-level guard counts and task-level four-
condition counts. Truncation, empty output and transport/timeout failures are
non-completions. Guard-produced insufficient-evidence responses are counted
separately and do not become legitimate completions merely because a source
reference says unknown. Ambiguous/missing references remain flagged and unresolved.
Pilot lexical extraction/attack screens remain separate from adjudicated quality;
unknown held-out target rubrics require review, with no guessed success label.
Actual distinct response first-visible/full-response timings are retained; cached
quality reuse is not additional latency evidence. The descriptive target-service
throughput excludes pacing/caching and is labelled accordingly; end-to-end/board
throughput remains null until actually measured.

## Native software backends for QAT finalists

`scripts/measure_frozen_software.py` now stages native execution of the frozen
U/B/C trained checkpoints as well as the original floating model. This is a
separate scored computation. An ordinary Hugging Face model receives the QAT
checkpoint's original trained FP32 weights, embeddings, LayerNorm, pooler and
classifier; the static W4/W8/A8 fake-quantization operations and their buffers are
omitted. Native FP32, CPU dynamic qint8, or supported CUDA FP16/BF16 autocast may
then run. This makes no native INT4 kernel claim and does not assert equality to
the ordered quantized detector.

Before Torch/model loading, the driver checks the real freeze/finalist and
checkpoint identity, exactly 16 integer W4/W8 group decisions, QAT metadata,
original training calibration/scale provenance and tokenizer checkpoint. At
runtime the native loader requires the full saved state inventory to equal the
ordinary HF state plus exactly 144 enumerated quantizer buffers: both weight
scales, input scale, zero point and activation/weight widths at all 24 sites.
Unknown or missing state is rejected. Every retained state tensor is checked for
name, shape, dtype and finite values; every trained parameter must be FP32.
After strict loading, the entire retained state is checked for exact values and
bytes. Static buffer shapes, widths and A8 scales are checked before omission.
The report explicitly records the trained-state inventory and omitted buffers.
These checks occur before dynamic qint8 conversion or device/autocast execution;
they do not claim optimized parameters remain byte-identical afterward.

The actual loader report becomes part of the observed native backend identity.
Each U/B/C backend/format requires its own complete temporary-threshold and
candidate-scoring development rows, an actual same-device/runtime floating FP32
reference, and the original FPR/recall protection gate before timing. All existing
split, source/freeze, run, window-membership, document-max, strict-decision,
workload/payload and measured-capacity/energy checks remain. Previously completed
floating backend records keep their existing identity fields and reference-reuse
route. A native QAT backend can fail protection even when its ordered quantized
finalist passed; such a failure does not produce a timing descriptor.

Only source/syntax/stdlib prerequisite previews were exercised for this change.
Evidence and prior source/diff are in
`results/goal7/native_qat_loader_source_v1/`. Four current role/format previews
remain blocked, load no model and read no held-out content. Runtime tensor
inventory/byte checks, backend development qualification and performance are
**unverified**, pending root release and an exclusive resource window. No
quantized numerical or score approval is granted by this loader.

After the actual finalist freeze is ready, focused verification starts with one
chosen role/format and the actual completed same-backend floating FP32 score
folder. These are future commands, not dispatched commands:

```powershell
# Source-only prerequisite preview; replace the reference folder with real evidence.
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/measure_frozen_software.py --stage qualify-backend --finalist-role U --backend cpu --format float32 --floating-reference /path/to/completed_cpu_floating_fp32
# After root resource release: performs strict native load checks and full development qualification.
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/measure_frozen_software.py --stage qualify-backend --finalist-role U --backend cpu --format float32 --floating-reference /path/to/completed_cpu_floating_fp32 --execute
# Only a resulting quality-passing native descriptor is eligible for actual timing.
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/measure_frozen_software.py --stage timing --backend-descriptor /path/to/actual_qualified_native_descriptor.json
```

The same qualification route accepts B/C, CPU `dynamic_qint8`, and CUDA
`float32`/`float16`/`bfloat16` where actually supported. Every role/format is
independently scored; available formats must be measured before selecting a
fastest supported protection-passing baseline. The timing command above is a
preview until separately released with `--execute`; no timing is implied here.

The coordinator separately adopted a conservative interpretation covering all
nine L128/256/512 by B1/8/32 cells: at least10000 completed checks per repeat,
three repeats, at both raw-text and prepared-input boundaries. B32 rounds upward
to10016 checks; B1/B8 use10000. Explicit lower-coverage software runs are
diagnostic only. This is the coordinator's adopted interpretation, not a claim
that the original wording uniquely prescribed all nine cells. The separate
source fix/evidence is in results/goal7/final_timing_grid_source_v1/ and remains
subject to its own review. Actual benchmarks remain unrun; native-loader source
acceptance grants no timing or numerical approval.

## Commands and remaining work

Run previews/checks from the repository or with absolute script paths:

```powershell
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/evaluate_frozen_finalists.py
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/evaluate_frozen_finalists.py --check-prerequisites
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/run_frozen_application.py
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/run_frozen_application.py --check-prerequisites
```

After the coordinator supplies the real development freeze and opens readiness:

```powershell
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/evaluate_frozen_finalists.py --prepare
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/evaluate_frozen_finalists.py --execute --phase quality --max-new-documents 32
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/evaluate_frozen_finalists.py --execute --phase challenge
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/evaluate_frozen_finalists.py --prepare-tableqa
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/evaluate_frozen_finalists.py --execute --phase extension --max-new-documents 32
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/run_frozen_application.py --prepare
& 'C:\research\GATE\.venv\Scripts\python.exe' -B scripts/run_frozen_application.py --execute --max-new-authorized-calls 0
```

These later commands are documented, not dispatched. Final numerical policy,
actual U/B/C selection/common evidence, freeze/readiness, the single pinned TableQA selection,
and any separately authorized held-out target generation remain prerequisites.
Goal 7's physical performance/load/energy grid, figures and actual final results
are separate unfinished work; no preparation output is a substitute for them.
