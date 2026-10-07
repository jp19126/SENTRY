# Goal 3 figure contract

The figures answer whether quantization changes detection after independently refitting the same nominal 1% benign calibration target. The claim is restricted to this fixed development split and one training seed. Final test remains closed; no interval across seeds or independent repeated training runs can be inferred.

Python/matplotlib is used throughout the existing Python workflow. The quantitative figures have one plot area each, at 183 mm width, with editable PDF/SVG text and PNG previews. Type is at least 7 pt. Their source JSON and generated CSV carry all plotted values; no observations or groups are excluded. Layout alignment across panels is not applicable to these single-panel figures.

1. Uniform precision: paired false-positive rates at the original floating threshold and each model's own refitted threshold, including pre-QAT and one-epoch QAT references. Attack misses and sample counts are supplied with source data/caption. A calibration target is not treated as the measured scoring FPR.
2. Single-group sensitivity: one W4 group in an otherwise W8 encoder, paired false-positive rates at a common W8 reference threshold and after each candidate's refit. All 16 groups remain in architectural order. Numerical-error and BCE rankings are reported separately in the sensitivity report, because perfect recall would make a correlation with recall loss undefined.
3. Available software costs: measured component means for the best tested CPU FP32/dynamic-INT8 setting and the saved matched CUDA FP32 pilot. Dynamic INT8 is a different numerical policy and its protection feasibility is labeled separately. These are short development timing pilots, not final throughput/energy results or FPGA comparisons.

Counts are actual documents, with related injected variants sharing source groups. Rates describe this fixed search set rather than independent-window samples. Timing component means use the saved 32 completed checks; no invented repeated-run error bars are added. The final evaluation's grouped uncertainty and 10,000-check repeats remain later-stage work.

Rendered figures will be inspected at final size. Source, PDF text and collision checks concern export quality only and do not establish scientific validity.
