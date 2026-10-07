# Goal 1 data preparation

Prepared 2026-09-26T02:48:07.812525+00:00 with seed 42. Labels mean **injected-instruction presence**, never whether an LLM followed it. No target LLM or paid judge was called.

## Sources and scope

- [BIPIA official builder](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/bipia/data/base.py), [EmailQA prompts](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/bipia/data/email.py), [insertion rules](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/bipia/data/utils.py), [evaluation registry](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/bipia/metrics/regist.py); revision `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`. The builder forms context × attack × insertion-position combinations and preserves the legitimate question/reference. Its response judges include model-based semantic checks and fuzzy matching; neither is a presence label. We retain exact original harmless payloads and the start/end newline-insertion functions. Middle insertion and stealth encoding are outside this initial pool.
- [AESLC authors' corpus](https://github.com/ryanzhumich/AESLC/tree/7afc087a1cc234d07121f0d4a2d87102ddceabc2), revision `7afc087a1cc234d07121f0d4a2d87102ddceabc2`: Enron email bodies supply adequate distinct detector-training backgrounds. Bodies precede the `@subject` delimiter; subjects/annotations are not detector input. Its CC BY-NC-SA 4.0 terms and Zhang & Tetreault (ACL 2019) attribution apply. Clean historical emails are operational negatives for injection presence, not labels for harmful topics or spam. This is an explicitly constructed BIPIA-style detector dataset, not an official BIPIA release or an AESLC injection benchmark.
- [NotInject official files](https://github.com/leolee99/PIGuard/tree/1b5751e88bf7475acbedfc8eda795ce060307c84/datasets), revision `1b5751e88bf7475acbedfc8eda795ce060307c84`: all 339 rows remain a separate benign challenge, including original trigger-word/category metadata. Multilingual challenge cases are retained and identified, outside the main English-task scope. No challenge row enters fitting or selection.
- Original EmailQA upstream invoices have only 100 rows as well ([OpenAI Evals immutable snapshot](https://github.com/openai/evals/blob/8eac7a7de5215c907fbddc30efdaf316913eccdd/evals/registry/data/invoices/match.jsonl)); expanding independent EmailQA pilot tasks from that file is not possible.

## Fixed source grouping and splits

Original relative filenames/line numbers are source IDs. Whitespace-normalized exact body equality joins source copies across files, official splits and datasets; all related injection variants stay with that group. This does not claim fuzzy/semantic near-duplicate detection. Raw source snapshots preserve original text, question and reference answers. BIPIA task records preserve repeated questions/answers even when detector documents are collapsed to one clean body.

Any official test occurrence reserves the full group for test; AESLC official dev reserves its full group for final calibration. Thus 11 BIPIA train/test shared exact bodies remain held out. Remaining AESLC official train groups allocate 80% training and 20% search, with search split equally into temporary threshold and candidate scoring groups. The 34 remaining BIPIA official-train body groups allocate 8 final calibration, 6 temporary threshold and 20 candidate scoring, giving a usable application pilot without using its emails for weight updates. This source-specific arrangement supersedes the fallback 60/20/10/10 fractions because suitable official splits exist.

| Partition | Source groups / distinct benign bodies | Clean documents | Injected variants |
|---|---:|---:|---:|
| train | 10790 | 10790 | 21580 |
| search_temporary_threshold | 1355 | 1355 | 2710 |
| search_candidate_scoring | 1369 | 1369 | 2738 |
| final_calibration | 1717 | 1717 | 3434 |
| test | 1810 | 1810 | 3620 |

Every detector source group has one clean document and two generated variants: one start injection and one end injection, with payloads sampled reproducibly without replacement within each body. Development uses 55 official train payloads (11 harmless categories); test uses 50 official test payloads (10 harmless categories). Official held-out payloads remain test-only. This limited task-diversion/answer-transformation pool excludes malware promotion, scams and misinformation; results cannot claim coverage of every injection type.

## Application pairs and benign-count limitation

`data/prepared/development_pairs.jsonl` contains **100 clean/attacked pairs but only 20 distinct source emails and 20 distinct source/question tasks**. Five official harmless payloads per email create the pairs; clean sides repeat and do not become additional independent documents. All pilot groups belong to candidate scoring, never training, temporary threshold, final calibration or test. Each pair preserves the original question/reference answer and exact injection span plus a fixed human-review success rubric. The five representative pairs are shown in `reports/representative_pairs.md`. Each EmailQA case naturally supplies one chunk; none is duplicated to manufacture five chunks.

The held-out pool has **1810 distinct benign test bodies**, short of the 3,000 aim by **1190**. No duplicated clean side, attack variant, window or NotInject row is counted toward this number. EmailQA-only test has 44 groups; its FPR resolution alone is coarse. Report AESLC/BIPIA strata separately as well as any declared aggregate; Enron calibration does not establish a precise EmailQA-domain 1% FPR guarantee. Broader final-test acquisition remains a later evidence need, not a reason to delay early development.

## Outputs and check

`data/prepared/{train,search_temporary_threshold,search_candidate_scoring,final_calibration,test}.jsonl`, `development_pairs.jsonl`, `notinject.jsonl` and `data/prepared/summary.json`. Detector inputs use `text`; clean labels are 0 and injection labels 1; `injection_spans` are half-open character intervals. Overlapping token-window supervision must use these spans, and final scores/FPR use document aggregation.

The required fields/labels, original-group and normalized-body split separation, exact inserted spans, pilot placement and NotInject exclusion checks passed during this preparation. Only these prescribed checks ran; there is no hash/integrity framework. No detector training or target inference occurred.

Reproduce from any working directory:
```powershell
& 'C:\research\GATE\.venv\Scripts\python.exe' '\\wsl.localhost\Ubuntu\home\jp19126\Projects\SENTRY\scripts\prepare_data.py'
```
Already acquired immutable raw snapshots are reused. Settings are read from `configs/project.json`.
