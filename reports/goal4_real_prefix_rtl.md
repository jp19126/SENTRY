# Saved-input RTL prefix

All six calls passed for the existing saved W8-QAT L256 input. Three invocations each process two distinct 128-token halves; these are not repetitions. Captured RTL bytes feed each subsequent operation.

| Operation | Comparison over 65,536 values | Cycles per half |
|---|---|---:|
| Embedding addition | Bit-identical to separate FP32 sums | 98,455 |
| Embedding LayerNorm | Maximum absolute error 9.5367431640625e-7; 12,415 bit differences; zero failures under existing atol/rtol 2e-4/2e-4 | 208,648 |
| Layer0-query A8 | Zero code differences versus saved ordered CPU codes and same-input quantization of captured RTL LN output | 98,484 |

The arithmetic prefix totals **811,174 cycles (4.055870 ms at 5 ns)** under the existing bounded six-port memory model. Python orchestration and file handoffs are outside simulated cycles. This is a prefix numerical diagnostic, not whole-detector execution, general protection acceptance, or board timing.

The successful synthesis record, current source and testbench copies, every reachable generated RTL copy, and existing compiled snapshot were checked before execution. No RTL, testbench, compilation, synthesis, model forward, or dataset scoring changed. Evidence, exact references, commands, logs and captured bytes are in `results/goal4/numerical_bridge/rtl_prefix_v1/`; `run.json` records the source/checkpoint/case associations and confirms no owned process remains.

Real-input simulation time differed from earlier synthetic estimates. The unchanged 900-second limit applied separately to each stage; all completed without timeout, extension, or rerun. Observed simulator CPU seconds were: embedding_add 142.74, embedding_ln 681.79, layer0_query_a8 397.87. These host simulation times are not device latency.

Driver: `scripts/run_real_prefix_rtl.py`. Default is offline preview. The executed command was `wsl.exe -d Ubuntu --exec /usr/bin/python3 /home/jp19126/Projects/GATE/scripts/run_real_prefix_rtl.py --execute`. Existing evidence is never overwritten by execution, and the existing pause marker is respected.
