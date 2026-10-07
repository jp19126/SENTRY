"""Plot Goal 3 development evidence from saved results, without model inference."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "goal3"
OUT = ROOT / "results" / "goal3" / "figures"
COLORS = {"fixed": "#697984", "refit": "#167A89", "target": "#A14545"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.titlesize": 10, "axes.titleweight": "normal",
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "legend.frameon": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table(name, rows):
    with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    fig.canvas.draw()
    # Each figure has exactly one plot area; there are no comparable panel edges.
    (OUT / (name + ".alignment.json")).write_text(json.dumps({
        "verdict": "NOT APPLICABLE", "plot_areas": len(fig.axes),
        "reason": "single-panel scientific figure", "width_mm": 183,
    }, indent=2) + "\n")
    fig.savefig(OUT / (name + ".svg"))
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".png"), dpi=300)
    fig.savefig(OUT / (name + ".tiff"), dpi=600)
    plt.close(fig)


def uniform():
    fp = read(SOURCE / "floating_reference.json")["metrics"]
    p = fp["primary"]["candidate_scoring"]
    rows = [{"model": "FP32", "fixed_threshold": p["threshold"],
             "refit_threshold": p["threshold"], "fixed_fpr_pct": p["fpr"] * 100,
             "refit_fpr_pct": p["fpr"] * 100, "fixed_fn": p["fn"],
             "refit_fn": p["fn"], "benign": p["benign"], "attacks": p["attacks"]}]
    for name, label in [("ptq_w8_a8", "W8A8 PTQ"), ("ptq_w4_a8", "W4A8 PTQ"),
                        ("qat_w8_a8", "W8A8 QAT"), ("qat_w4_a8", "W4A8 QAT")]:
        metrics = read(SOURCE / name / "candidate.json")["metrics"]
        refit = metrics["primary"]["candidate_scoring"]
        fixed = metrics["at_floating_threshold"]
        rows.append({"model": label, "fixed_threshold": fixed["threshold"],
                     "refit_threshold": refit["threshold"],
                     "fixed_fpr_pct": fixed["fpr"] * 100, "refit_fpr_pct": refit["fpr"] * 100,
                     "fixed_fn": fixed["fn"], "refit_fn": refit["fn"],
                     "benign": refit["benign"], "attacks": refit["attacks"]})
    table("uniform_threshold_effects", rows)
    fig, ax = plt.subplots(figsize=(7.2047244, 3.7401575))
    fig.subplots_adjust(left=.12, right=.97, bottom=.20, top=.73)
    y = np.arange(len(rows))
    for index, row in enumerate(rows):
        ax.plot([row["fixed_fpr_pct"], row["refit_fpr_pct"]], [index, index], color="#C2C8CC", lw=1.3, zorder=1)
    ax.scatter([r["fixed_fpr_pct"] for r in rows], y, marker="s", s=32, c=COLORS["fixed"], label="Floating threshold", zorder=3)
    ax.scatter([r["refit_fpr_pct"] for r in rows], y, marker="o", s=32, c=COLORS["refit"], label="Own refitted threshold", zorder=4)
    ax.axvline(1, color=COLORS["target"], lw=.8, ls="--", label="1% FPR target")
    ax.set_yticks(y, [r["model"] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("Observed search false-positive rate (%)")
    ax.set_xlim(left=0); ax.grid(axis="x", alpha=.16)
    fig.suptitle("Uniform precision: threshold movement and detection", y=.96, fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.06), ncol=3, handletextpad=.5, columnspacing=1.2)
    attack_summary = " / ".join(str(r["refit_fn"]) for r in rows)
    fig.text(.12, .05, f"Refitted misses (row order): {attack_summary} of {p['attacks']:,} attacks each. Benign n = {p['benign']:,}.", fontsize=7)
    save(fig, "uniform_threshold_effects")


def sensitivity():
    changes = read(SOURCE / "single_group_sensitivity.json")
    w8 = read(SOURCE / "ptq_w8_a8" / "candidate.json")["metrics"]["primary"]
    w8_threshold = w8["threshold"]
    rows = []
    for change in changes:
        candidate = read(SOURCE / change["name"] / "candidate.json")
        p = candidate["metrics"]["primary"]["candidate_scoring"]
        # The paired comparison already uses W8's fixed threshold; recover counts
        # from its paired document transitions, not a second inference pass.
        delta = change["at_reference_threshold"]
        ref = w8["candidate_scoring"]
        fixed_fp = ref["fp"] + len(delta["new_false_positives"]) - len(delta["removed_false_positives"])
        fixed_fn = ref["fn"] + len(delta["new_misses"]) - len(delta["recovered_attacks"])
        rows.append({"group": change["group"], "fixed_threshold": w8_threshold,
                     "refit_threshold": p["threshold"], "fixed_fpr_pct": fixed_fp / p["benign"] * 100,
                     "refit_fpr_pct": p["fpr"] * 100, "fixed_fp": fixed_fp, "refit_fp": p["fp"],
                     "fixed_fn": fixed_fn, "refit_fn": p["fn"], "benign": p["benign"], "attacks": p["attacks"],
                     "bce_change": change["mean_document_bce_change"], "window_logit_mse": change["window_logit_mse"]})
    table("group_sensitivity", rows)
    fig, ax = plt.subplots(figsize=(7.2047244, 5.7086614))
    fig.subplots_adjust(left=.27, right=.97, bottom=.14, top=.83)
    y = np.arange(len(rows))
    for index, row in enumerate(rows):
        ax.plot([row["fixed_fpr_pct"], row["refit_fpr_pct"]], [index, index], color="#C2C8CC", lw=1.1, zorder=1)
    ax.scatter([r["fixed_fpr_pct"] for r in rows], y, marker="s", s=26, c=COLORS["fixed"], label="W8 threshold", zorder=3)
    ax.scatter([r["refit_fpr_pct"] for r in rows], y, marker="o", s=26, c=COLORS["refit"], label="Own refitted threshold", zorder=4)
    ax.axvline(1, color=COLORS["target"], lw=.8, ls="--", label="1% FPR target")
    ax.set_yticks(y, [r["group"].replace("layer", "L").replace(".", " / ").replace("attention_output", "Attention out").replace("ffn_input", "FFN in").replace("ffn_output", "FFN out").replace("qkv", "QKV") for r in rows])
    ax.invert_yaxis(); ax.set_ylim(len(rows)-.3, -.7)
    ax.set_xlabel("Observed search false-positive rate (%)"); ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=.16)
    fig.suptitle("Single W4 group in an otherwise W8A8 encoder", y=.97, fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.025), ncol=3, columnspacing=.9, handletextpad=.45)
    fig.text(.27, .04, "16 groups; one fixed development split and seed. PTQ only.\nAttack misses, paired changes and numerical errors are in the source CSV.", fontsize=7)
    save(fig, "group_sensitivity")


def software():
    profile = read(SOURCE / "software_profile" / "profile.json")
    if profile.get("status") != "complete":
        raise RuntimeError("Software profiling is incomplete; do not plot partial timing as final.")
    rows = []
    for key, label in [("float32", "CPU FP32"), ("dynamic_qint8", "CPU dynamic INT8")]:
        record = profile["formats"][key]
        if record["status"] != "measured":
            raise RuntimeError(f"Timing is not measured for {key}.")
        mean = record["best_timing"]["means"]
        rows.append({"platform": label, "threads": record["best_threads"],
                     "protection_feasible": record["meets_development_protection"], **mean})
    gpu = profile["cuda_float32_reused"]
    if "timing" not in gpu:
        raise RuntimeError("No matched saved GPU timing available.")
    rows.append({"platform": "CUDA FP32", "threads": "n/a", "protection_feasible": gpu["meets_development_protection"], **gpu["timing"]["means"]})
    # Use a union because host/device transfer columns are absent on CPU.
    keys = list(dict.fromkeys(key for row in rows for key in row))
    rows = [{key: row.get(key, 0) for key in keys} for row in rows]
    table("software_breakdown", rows)
    fig, ax = plt.subplots(figsize=(7.2047244, 3.7401575))
    fig.subplots_adjust(left=.25, right=.97, bottom=.20, top=.72)
    left = np.zeros(len(rows)); y = np.arange(len(rows))
    for field, label, color in [("preparation_ms", "Text preparation", "#BDC6CC"),
                                ("host_to_device_ms", "Host to device", "#BD985D"),
                                ("inference_and_aggregation_ms", "Inference + decision", "#167A89"),
                                ("result_transfer_ms", "Result transfer", "#526A83")]:
        values = np.array([r.get(field, 0) for r in rows])
        ax.barh(y, values, left=left, color=color, label=label, height=.55)
        left += values
    ax.set_yticks(y, [r["platform"] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("Mean time per completed check (ms)")
    ax.grid(axis="x", alpha=.16); ax.set_axisbelow(True)
    fig.suptitle("Available software: short timing pilot at 256 tokens, batch 1", y=.96, fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(.4, 1.07), ncol=2, columnspacing=1.8)
    flags = "; ".join(f"{r['platform']}: {'meets' if r['protection_feasible'] else 'fails'} search protection" for r in rows)
    fig.text(.05, .055, flags, fontsize=7)
    fig.text(.05, .018, "32 checks after 8 warm-ups. Best of 1/4/8 CPU threads. Dynamic INT8 differs from static W8A8.", fontsize=7)
    save(fig, "software_breakdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["quantization", "software", "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("quantization", "all"):
        uniform(); sensitivity()
    if args.stage in ("software", "all"):
        software()
    print(f"Saved {args.stage} figures from measured JSON in {OUT}")


if __name__ == "__main__":
    main()
