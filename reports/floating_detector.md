# Floating detector evidence — Goal 2

2026-09-26. Three floating-point epochs completed on the RTX 4070. The declared search selection rule (maximum document recall at the independently fitted temporary threshold, then minimum document binary cross-entropy, then earlier epoch) selects **epoch 3**. This is development evidence; final calibration and test files were not opened.

At the primary calibration target, epoch 3 detects **2,738/2,738** constructed injected search documents, but falsely flags **18/1,369 benign documents (1.3148%)**. This observed search FPR exceeds the 1% requirement; calibration itself has 13/1,355 exceedances (0.9594%). The threshold is **0.005549266934394836**, with strict `score > threshold`. No threshold adjustment or checkpoint-selection change was made after observing this gap. Later B/C feasibility must still meet the fixed requirement.

| Epoch | Training document-balanced window CE | Search document BCE | Search TP/FN | Search FP/TN | Threshold |
|---|---:|---:|---:|---:|---:|
| 1 | 0.09329382 | 0.00770011 | 2738/0 | 10/1359 | 0.0267638285 |
| 2 | 0.00720496 | 0.00625825 | 2738/0 | 12/1357 | 0.0261588246 |
| 3 | 0.00493260 | 0.00514651 | 2738/0 | 18/1351 | 0.0055492669 |

Selected primary-threshold document accuracy is 99.561724%; ordinary threshold-0.5 accuracy is 99.926954%. Training supervision uses exact injection spans: only intersecting windows are positive. Window CE is averaged within each document and then across each 32-document effective batch. A microbatch contains at most 32 windows, with variable gradient accumulation. AdamW uses learning rate 2e-5, weight decay .01, constant learning rate, seed 42, and exactly three epochs. Float32 parameters/activations/optimizer were used with TF32 disabled.

32,370 training documents produce 40,883 windows; 4,395 documents are long. All 254-content-token windows include CLS/SEP for total length 256 and overlap by 64 content tokens; document scores are window maxima. The first actual effective batch (included in epoch 1) has 32 documents/42 windows/two microbatches and uses 694,694,912 peak allocated bytes, 855,638,016 reserved bytes. No OOM or batch-size search was needed.

| Selected-model stratum | Documents | TP/FN | FP/TN | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| single_window | 3520 | 2338/0 | 10/1172 | 100.0000% | 0.8460% |
| long_document | 587 | 400/0 | 8/179 | 100.0000% | 4.2781% |
| source:AESLC | 4047 | 2698/0 | 18/1331 | 100.0000% | 1.3343% |
| source:BIPIA EmailQA | 60 | 40/0 | 0/20 | 100.0000% | 0.0000% |

| Calibration target | Threshold | Calibration FP/benign | Search TP/attacks | Search FP/benign |
|---|---:|---:|---:|---:|
| 0.5% | 0.0425027609 | 6/1355 | 2738/2738 | 6/1369 |
| 1.0% | 0.0055492669 | 13/1355 | 2738/2738 | 18/1369 |
| 2.0% | 0.0021699879 | 27/1355 | 2738/2738 | 34/1369 |

Nearby points reuse saved scores without new inference. The search sources are disjoint from detector training, but injected variants use the prepared BIPIA payload pool; perfect recall here does not establish robustness to unseen attack mechanisms. Only 20 EmailQA source groups occur in candidate scoring, so its separate small benign count cannot substantiate a precise domain-specific 1% FPR. NotInject and final test evidence belong to the final evaluation.

Five actual false-positive documents with scores, full text, source IDs and span metadata are saved in `results/goal2/search_error_examples.json`; there are no false-negative examples in this selected search run. Example false-positive IDs and risks:

- `aeslc:train:beck-s_inbox_490.subject:clean`: 0.99796969.
- `aeslc:train:buy-r_inbox_622.subject:clean`: 0.00594725.
- `aeslc:train:geaccone-t_inbox_284.subject:clean`: 0.00726354.
- `aeslc:train:germany-c_inbox_173.subject:clean`: 0.03136212.
- `aeslc:train:germany-c_sent_29.subject:clean`: 0.00829681.

The model contains **11,171,074 parameters**, including **7,945,728 embedding-module parameters**, totaling **44,684,296 bytes** of float32 parameters. Epoch 3 `model.safetensors` is 44,692,608 bytes including file metadata. `training_run.json` records every parameter shape, each encoder matrix shape, exact package versions, configuration and device. The saved label mapping is benign=0/injection=1.

The short timing pilot runs 8 warm-up and 32 measured single-window checks, batch 1 with total padded length 256. CUDA synchronization bounds each phase; this is not a steady-state final workload benchmark. Mean times:

| Phase | Mean milliseconds |
|---|---:|
| preparation_ms | 0.500547 |
| host_to_device_ms | 0.181972 |
| inference_and_aggregation_ms | 5.100134 |
| result_transfer_ms | 0.053753 |
| total_ms | 5.836406 |

All three model/tokenizer checkpoints and optimizer/RNG states remain in native `C:\research\GATE\checkpoints\floating\epoch_1` through `epoch_3`; selected checkpoint is `epoch_3`, also copied with its optimizer/RNG state to canonical `checkpoints/floating/epoch_3` for subsequent work. The full native results are mirrored into the canonical repository under `results/goal2/`. `pilot_guard_scores.jsonl` contains all 200 development pair-side IDs for the application runner; `search_threshold.json` supplies its unchanged development threshold.

Execution note: after epoch 1, a report expected a `task` field instead of the actual `source_dataset` field. The reporting code was corrected, both saved search score files were reused, and training resumed at epoch 2 from the saved model/optimizer/RNG state. No epoch or inference was repeated to fix the report. Epoch 1 whole-epoch peak memory was not persisted; it remains null. Its first-batch measured memory evidence is preserved. Future checkpoints persist pending evaluation status before scoring.

Reproduction/resume command: `C:\research\GATE\.venv\Scripts\python.exe C:\research\GATE\scripts\train_detector.py --microbatch-windows 32 --resume`. A completed run reuses its artifacts. For an authorized pause, create native `results/goal2/pause.request`; the running trainer saves its current optimizer/RNG after the current effective batch.

Prompt Guard access and the Gemini application experiment are tracked separately; this detector report does not claim their results or any FPGA, quantized arithmetic, power or final-test measurement.

