# CPU/GPU software pilot — Goal 3

2026-09-26. Short matched-input software timing is complete. CPU float32 at eight threads is faster than the actual dynamic-int8 route tested here. **Neither CPU format meets the fixed 1% observed development FPR constraint**, so the timing results are measurements of these routes, not feasible protection-matched winners.

Both formats use selected floating epoch 3 and all 24 BERT encoder linear modules. CPU float32 remains float32 throughout. Dynamic quantization uses actual PyTorch `torch.qint8` packed weights and runtime activation quantization in those 24 modules; embeddings, classifier and nonlinear operations remain float32. The actual and sole supported quantized engine in this Windows build is **oneDNN**. The installed PyTorch dynamic Linear version 4 calls `quantized.linear_dynamic(..., reduce_range=True)`. This format differs from the project's fixed-scale signed W8A8 arithmetic, and no INT4 execution time is claimed.

Timing uses the same 40 real single-window candidate-scoring documents as the Goal 2 GPU pilot: eight warm-up and 32 measured checks, padded total length 256, batch 1. CPU inter-op threads are fixed to one. Intra-op counts 1, 4 and 8 were tried once for each format; lowest mean raw-text-to-decision time selected the count. Timing ran only after the quantization worker released the host/GPU. Later quality scoring overlapped report/plot work but its duration is not used as a performance result.

| CPU format | Threads | Preparation ms | Inference/aggregation ms | Total ms |
|---|---:|---:|---:|---:|
| float32 | 1 | 0.542634 | 21.659319 | 22.204125 |
| float32 | 4 | 0.528781 | 9.659088 | 10.189713 |
| float32 | 8 | 0.611644 | 7.863903 | 8.477506 |
| dynamic_qint8 | 1 | 0.532778 | 17.019537 | 17.554103 |
| dynamic_qint8 | 4 | 0.526978 | 9.172700 | 9.701419 |
| dynamic_qint8 | 8 | 0.673953 | 9.492028 | 10.167813 |

The matched saved CUDA float32 pilot is reused: preparation 0.500547 ms, host-to-device transfer 0.181972 ms, inference/aggregation 5.100134 ms, result transfer 0.053753 ms, total **5.836406 ms**. Its input document IDs and window length match. Goal 2 used CUDA synchronization at each boundary and disabled TF32. No new GPU benchmark or final 10,000-check campaign was run.

Protection was measured on complete documents using each CPU format's own temporary threshold fitted only to the assigned 1,355 benign search-threshold documents. Both CPU formats were scored at **window batch 1**, matching the timed path; this is necessary because dynamic activation quantization can depend on the inference batch. Candidate scoring contains 2,738 injected and 1,369 benign documents.

| CPU format | Temporary threshold | Calibration FP/benign | Candidate TP/FN | Candidate FP/TN | Recall | FPR | Feasible |
|---|---:|---:|---:|---:|---:|---:|---|
| float32 | 0.005549295805 | 13/1355 | 2738/0 | 18/1351 | 100.0000% | 1.3148% | False |
| dynamic_qint8 | 0.003368974663 | 13/1355 | 2738/0 | 21/1348 | 100.0000% | 1.5340% | False |

CPU float32 has the same 18 false positives as the saved GPU reference, with only the expected small floating-point threshold difference. Dynamic-int8 has 21 false positives; its long-document count is 9/187 (4.8128%), versus 12/1,182 (1.0152%) for single-window benign documents. Its document BCE is 0.00330353, lower than the floating reference despite a worse low-FPR operating point; mean loss alone does not certify the protection requirement. All dynamic-int8 false positives occur in AESLC; the separate EmailQA subset has only 20 benign sources and no false positives.

Nearby 0.5%/2% points, full source/length strata, raw window logits/scores and document aggregates are saved under `results/goal3/software_profile/{float32,dynamic_qint8}/`. They reuse the same inference scores. No final-calibration, final-test or NotInject data were opened by this profiler.

Evidence: `profile.json` records configuration, package versions, chosen threads, actual quantized module names/shapes/dtypes/qschemes, the backend, all timing samples, and protection flags. The six `*_threads_*.json` files contain raw timing records; `reports/goal3_software_profile.log` records execution. PyTorch issued an API deprecation warning but its installed dynamic quantized kernels executed successfully. No extra framework was installed.

Reproduce/reuse: `C:\research\GATE\.venv\Scripts\python.exe C:\research\GATE\scripts\profile_software.py`. The completed profile is reused without new measurements. The source also exists in the canonical repository, where the selected epoch-3 checkpoint is available. This is a short optimization pilot, not final repeated latency, throughput, sustained-load or energy evidence.
