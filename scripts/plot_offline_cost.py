"""Plot the saved Goal 4 whole-check analytical cost sensitivity (no inference)."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/goal4/whole_check_anchor_sensitivity_review_v3.json"
V1_OUT = ROOT / "results/goal4/figures/offline_cost_v1"
V2_OUT = ROOT / "results/goal4/figures/offline_cost_v2"
OUT = V1_OUT


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(record: dict) -> list[dict]:
    scenarios = {row["identity"]: row for row in record["scenarios"]}
    rows = []
    for anchor in record["anchors"]:
        scenario = scenarios[anchor["scenario"]]
        terms = anchor["terms"]
        calculated = sum(terms[key] for key in (
            "internal_ms", "host_preprocessing_ms", "uart_transport_ms",
            "supervisor_sync_ms", "additional_memory_wait_ms"))
        if not math.isclose(calculated, anchor["estimated_complete_checking_ms"], rel_tol=0, abs_tol=1e-8):
            raise ValueError("Source cost components do not sum to the saved total")
        rows.append({
            "scenario": anchor["scenario"],
            "additional_read_response_cycles": scenario["additional_read_response_cycles"],
            "additional_write_response_cycles": scenario["additional_write_response_cycles"],
            "supervisor_sync_ms_assumption": scenario["supervisor_sync_ms"],
            "weight_bits": anchor["weight_bits"],
            "profile_id": anchor["profile_id"],
            "estimated_complete_checking_ms": anchor["estimated_complete_checking_ms"],
            "integrated_feasible": anchor["integrated_feasible"],
            "internal_ms": terms["internal_ms"],
            "host_preprocessing_ms": terms["host_preprocessing_ms"],
            "uart_transport_ms": terms["uart_transport_ms"],
            "supervisor_sync_ms": terms["supervisor_sync_ms"],
            "additional_memory_wait_ms": terms["additional_memory_wait_ms"],
            "additional_memory_read_requests": terms["additional_memory_read_requests"],
            "additional_memory_write_requests": terms["additional_memory_write_requests"],
        })
    return rows


def validate_frozen_cost_table(path: Path, table: dict, validation: dict,
                               source_record: dict, rows: list[dict]) -> dict:
    expected_caps = [7624.0, 7642.0, 7660.0, 7679.0]
    if (table.get("schema") != "sentry_goal4_frozen_complete_cost_table_v1"
            or table.get("latency_unit") != "ms"
            or table.get("latency_scope") != "full_detector_host_to_decision"
            or table.get("detector_execution") != "fpga"
            or table.get("primary_length") != 256 or table.get("primary_batch") != 1
            or table.get("frozen_latency_caps_ms") != expected_caps):
        raise ValueError("The supplied frozen table does not match the approved L256/B1 cost-table scope")
    if (validation.get("state") != "actual_cost_sources_validated"
            or validation.get("ready") is not True or validation.get("blockers") != []
            or validation.get("caps_ms") != expected_caps
            or validation.get("sensitivity_scenarios") != 27
            or validation.get("physical_board_activity") is not False):
        raise ValueError("Actual cost-table validation is incomplete or out of scope")
    if table.get("physical_board_measurement") is not False:
        raise ValueError("Physical calibration status must remain explicitly unmeasured")
    profile_ids = {"lanes16", "lanes32", "lanes64"}
    profiles = {p["id"]: p for p in table.get("profiles", [])}
    if set(profiles) != profile_ids:
        raise ValueError("Expected the three common L256 profiles in the frozen table")
    validated_profiles = validation.get("profiles", {})
    closures = table.get("closure_associations", {})
    if set(closures) != profile_ids or set(validated_profiles) != profile_ids:
        raise ValueError("Validated profile and closure associations must cover all three profiles")
    for profile in sorted(profile_ids):
        p = profiles[profile]
        validated = validated_profiles[profile]
        association = closures[profile]
        for field in ("integrated_eligible", "timing_met", "bandwidth_feasible"):
            if p.get(field) is not True or association.get("record", {}).get(field) is not True:
                raise ValueError(f"Frozen table does not establish {field} for {profile}")
        if validated.get("validated") is not True or validated.get("integrated_eligible") is not True:
            raise ValueError(f"Validation report does not establish offline closure eligibility for {profile}")
        for version in ("record", "reparsed"):
            evidence = association.get(version, {})
            if (evidence.get("profile_id") != profile or evidence.get("length") != 256
                    or evidence.get("integrated_eligible") is not True
                    or evidence.get("timing_met") is not True
                    or evidence.get("bandwidth_feasible") is not True):
                raise ValueError(f"Actual and reparsed closure evidence is incomplete for {profile}")
        if association.get("path") != p.get("closure_report"):
            raise ValueError(f"Closure association differs from frozen profile path for {profile}")
        actual_closure = read_json(ROOT / association["path"])
        if (actual_closure.get("profile_id") != profile or actual_closure.get("length") != 256
                or actual_closure.get("integrated_eligible") is not True
                or actual_closure.get("timing_met") is not True
                or actual_closure.get("bandwidth_feasible") is not True):
            raise ValueError(f"Current actual closure report is not eligible for {profile}")

    anchor_name = {4: "uniform_w4a8", 8: "uniform_w8a8"}
    nominal_assumptions = {"additional_read_response_cycles": 16,
                          "additional_write_response_cycles": 16,
                          "supervisor_sync_ms": 1.0}
    nominal = {(x["anchor"], x["profile_id"]): x["estimated_complete_checking_ms"]
               for x in table.get("nominal_sensitivity", [])
               if x.get("assumptions") == nominal_assumptions}
    source_nominal = [r for r in source_record["anchors"] if r["scenario"] == "nominal"]
    if len(nominal) != 6 or len(source_nominal) != 6:
        raise ValueError("Expected all six frozen nominal profile/precision values")
    for row in source_nominal:
        key = (anchor_name[row["weight_bits"]], row["profile_id"])
        if nominal.get(key) != row["estimated_complete_checking_ms"]:
            raise ValueError(f"Frozen nominal cost differs from sensitivity source at {key}")

    # The five earlier scenarios are retained verbatim as a declared subset of
    # the frozen table's 27-point read/write/synchronization Cartesian product.
    assumptions_by_name = {
        x["identity"]: {"additional_read_response_cycles": x["additional_read_response_cycles"],
                        "additional_write_response_cycles": x["additional_write_response_cycles"],
                        "supervisor_sync_ms": x["supervisor_sync_ms"]}
        for x in source_record["scenarios"]}
    table_rows = {(x["anchor"], x["profile_id"],
                   x["assumptions"]["additional_read_response_cycles"],
                   x["assumptions"]["additional_write_response_cycles"],
                   x["assumptions"]["supervisor_sync_ms"]): x["estimated_complete_checking_ms"]
                  for x in table.get("nominal_sensitivity", [])}
    if len(table_rows) != 162:
        raise ValueError("Frozen sensitivity table must contain all 2x3x27 entries")
    for row in rows:
        key = (anchor_name[row["weight_bits"]], row["profile_id"],
               *assumptions_by_name[row["scenario"]].values())
        if table_rows.get(key) != row["estimated_complete_checking_ms"]:
            raise ValueError(f"Source sensitivity value is not an exact frozen-table subset entry: {key}")
    return {
        "cost_table": str(path.relative_to(ROOT)),
        "validation_report": str((ROOT / table["validation_report"]).relative_to(ROOT)),
        "validation_state": validation["state"],
        "caps_seconds": [x / 1000 for x in expected_caps],
        "cartesian_scenarios": 27,
        "displayed_scenarios": len(source_record["scenarios"]),
        "displayed_scenarios_are_non_exhaustive_subset": True,
        "validated_profile_closures": sorted(profile_ids),
        "physical_board_measurement": False,
        "physical_calibration_pending": True,
    }


def validate(record: dict, rows: list[dict], cost_table: dict | None = None) -> None:
    if len(rows) != 30:
        raise ValueError(f"Expected all 30 saved sensitivity rows, found {len(rows)}")
    expected = {(p, w, s) for p in ("lanes16", "lanes32", "lanes64")
                for w in (4, 8) for s in [x["identity"] for x in record["scenarios"]]}
    actual = {(r["profile_id"], r["weight_bits"], r["scenario"]) for r in rows}
    if actual != expected:
        raise ValueError("Sensitivity rows do not cover the full 3 profile x 2 precision x 5 scenario grid")
    if cost_table is None:
        if record.get("caps") is not None:
            raise ValueError("This historical source is expected to leave latency caps unset")
        if record.get("cached32_closure_pending") is not True or record.get("physical_measurement") is not False:
            raise ValueError("Historical source state changed; review its status before plotting")
    canonical_order = None
    for scenario, ranked in record["rankings"].items():
        measured = sorted((r for r in rows if r["scenario"] == scenario),
                          key=lambda r: r["estimated_complete_checking_ms"])
        measured_order = [(r["weight_bits"], r["profile_id"]) for r in measured]
        saved_order = [(r["weight_bits"], r["profile_id"]) for r in ranked]
        if saved_order != measured_order:
            raise ValueError("Saved ranking differs from row-level sensitivity data")
        if canonical_order is None:
            canonical_order = saved_order
        elif saved_order != canonical_order:
            raise ValueError("Configuration ranking changes across the reviewed scenarios")
    if cost_table is None and any(r["integrated_feasible"] is not None for r in rows if r["profile_id"] == "lanes32"):
        raise ValueError("Historical source does not establish cached32 integrated eligibility")


def export_source_csv(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "offline_cost_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(rows: list[dict], cost_context: dict | None = None) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from audit_panel_alignment import matplotlib_layout_manifest, require_matplotlib_panel_alignment

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6.3, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    config_colors = {"lanes16": "#718096", "lanes32": "#23808A", "lanes64": "#263E59"}
    cost_colors = {
        "internal_ms": "#8295A4", "host_preprocessing_ms": "#C9D2D8",
        "uart_transport_ms": "#AAB6BF", "supervisor_sync_ms": "#D8B98B",
        "additional_memory_wait_ms": "#D97947",
    }
    components = list(cost_colors)
    labels = {"internal_ms": "Internal service", "host_preprocessing_ms": "Host preprocessing",
              "uart_transport_ms": "UART transfer", "supervisor_sync_ms": "A72 sync allowance",
              "additional_memory_wait_ms": "Additional memory waits"}
    profiles = ["lanes16", "lanes32", "lanes64"]
    config_order = [(p, w) for p in profiles for w in (8, 4)]
    scenario_order = ["low", "nominal", "read_low_write_high", "read_high_write_low", "high"]
    scenario_labels = ["Low\n4/4; 0.1 ms", "Nominal\n16/16; 1 ms",
                       "Read-low /\nwrite-high\n4/40; 1 ms",
                       "Read-high /\nwrite-low\n40/4; 1 ms", "High\n40/40; 2 ms"]
    by_key = {(r["profile_id"], r["weight_bits"], r["scenario"]): r for r in rows}

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.2047, 6.75), sharex=False)
    fig.subplots_adjust(left=0.205, right=0.985, top=0.835, bottom=0.22, hspace=0.56)
    fig.suptitle("Analytical warm-window cost sensitivity at L256/B1", fontsize=11, y=0.985)
    ax0.set_title("Nominal warm-window estimate: additional memory waits dominate", fontsize=8.3, pad=9, loc="left")
    y = np.arange(len(config_order))
    left = np.zeros(len(config_order))
    nominal_rows = []
    for profile, bits in config_order:
        nominal_rows.append(by_key[(profile, bits, "nominal")])
    for component in components:
        values = np.array([r[component] / 1000 for r in nominal_rows])
        ax0.barh(y, values, left=left, height=0.62, color=cost_colors[component],
                 edgecolor="white", linewidth=0.35, label=labels[component])
        left += values
    for idx, total in enumerate(left):
        ax0.text(total + 0.08, idx, f"{total:.2f} s", va="center", ha="left", fontsize=6.2)
    ax0.set_yticks(y, [f"{p.replace('lanes','')} lanes · W{bits}" for p, bits in config_order])
    ax0.invert_yaxis()
    ax0.set_xlim(0, float(left.max()) * 1.23)
    ax0.set_xlabel("Estimated complete warm-window cost (s)")
    ax0.grid(False)
    legend_handles, legend_labels = ax0.get_legend_handles_labels()
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.58, 0.945),
               ncol=3, frameon=False, columnspacing=1.2, handlelength=1.2)
    ax0.annotate("a", xy=(0, 1), xycoords="axes fraction", xytext=(-19, 4),
                 textcoords="offset points", fontsize=8, fontweight="bold", annotation_clip=False)

    panel_b_title = ("Five of 27 wait scenarios: absolute costs move; ranking is unchanged"
                     if cost_context else "Assumption sensitivity: absolute costs move; ranking is unchanged")
    ax1.set_title(panel_b_title, fontsize=8.3, pad=9, loc="left")
    x = np.arange(len(scenario_order))
    for profile, bits in config_order:
        vals = [by_key[(profile, bits, scenario)]["estimated_complete_checking_ms"] / 1000
                for scenario in scenario_order]
        ax1.plot(x, vals, color=config_colors[profile], linestyle="-" if bits == 8 else "--",
                 marker="o" if bits == 8 else "s", markersize=3.3, linewidth=1.2,
                 label=f"{profile.replace('lanes','')} lanes · W{bits}")
    ax1.set_xticks(x, scenario_labels)
    ax1.set_ylabel("Estimated complete warm-window cost (s)")
    ax1.set_xlabel("Additional read/write response cycles; labels also show A72 sync allowance (ms)")
    ax1.grid(axis="y", color="#D9DEE2", linewidth=0.5, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", ncol=3, frameon=False, columnspacing=1.0, handlelength=1.7)
    ax1.annotate("b", xy=(0, 1), xycoords="axes fraction", xytext=(-19, 4),
                 textcoords="offset points", fontsize=8, fontweight="bold", annotation_clip=False)
    if cost_context:
        fig.text(0.205, 0.073,
                 "Frozen common caps (s): 7.624 / 7.642 / 7.660 / 7.679. Estimates are analytical warm-window costs; physical calibration pending.",
                 fontsize=6.0, ha="left", va="center")
        fig.text(0.205, 0.039,
                 "All three L256 profiles passed offline closure. Five shown assumptions are a subset of the 27 Cartesian scenarios, not exhaustive.",
                 fontsize=6.0, ha="left", va="center")
    else:
        fig.text(0.205, 0.055,
                 "Low-to-high assumptions can reverse pass/fail against a cap within each range; no cap is adopted. "
                 "Cached32 closure remains pending. Scenarios are engineering assumptions, not confidence intervals.",
                 fontsize=6.2, ha="left", va="center")

    fig.canvas.draw()
    layout = matplotlib_layout_manifest(fig, panel_ids=["a", "b"])
    (OUT / "offline_cost.alignment-layout.json").write_text(
        json.dumps(layout, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    alignment_path = OUT / "offline_cost.alignment.json"
    require_matplotlib_panel_alignment(
        fig, json_out=alignment_path, overlay_svg=OUT / "offline_cost.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
    )
    fig.savefig(OUT / "offline_cost.svg", bbox_inches="tight")
    fig.savefig(OUT / "offline_cost.pdf", bbox_inches="tight")
    fig.savefig(OUT / "offline_cost.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "offline_cost.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-table", type=Path,
                        help="actual frozen Goal 4 table; creates fresh offline_cost_v2 output")
    args = parser.parse_args()
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    if not skill_scripts.is_dir():
        raise RuntimeError("Selected Python figure alignment helper is unavailable")
    sys.path.insert(0, str(skill_scripts))
    record = read_json(SOURCE)
    rows = build_rows(record)
    validate(record, rows)
    cost_context = None
    if args.cost_table:
        cost_path = args.cost_table if args.cost_table.is_absolute() else ROOT / args.cost_table
        cost_path = cost_path.resolve()
        if not cost_path.is_file():
            raise FileNotFoundError(f"Frozen cost table not found: {cost_path}")
        OUT = V2_OUT
        table = read_json(cost_path)
        validation_path = ROOT / table.get("validation_report", "")
        if not validation_path.is_file():
            raise FileNotFoundError("Frozen cost table validation report is missing")
        validation = read_json(validation_path)
        cost_context = validate_frozen_cost_table(cost_path, table, validation, record, rows)
    export_source_csv(rows)
    if cost_context:
        v1_csv = V1_OUT / "offline_cost_source.csv"
        v2_csv = OUT / "offline_cost_source.csv"
        if not v1_csv.is_file() or v1_csv.read_bytes() != v2_csv.read_bytes():
            raise ValueError("v2 must preserve the complete v1 30-row source CSV unchanged")
    render(rows, cost_context)
    summary = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_status": record["status"],
        "rows_preserved": len(rows),
        "scenario_count": len(record["scenarios"]),
        "historical_source_caps": record["caps"],
        "historical_cached32_pending": record["cached32_closure_pending"],
        "physical_measurement": record["physical_measurement"],
        "rank_order_by_scenario": {
            scenario: [f"{entry['profile_id']}/W{entry['weight_bits']}" for entry in ranked]
            for scenario, ranked in record["rankings"].items()
        },
        "current_frozen_cost_context": cost_context,
        "claim_scope": "Analytical L256/B1 warm-window cost scenarios only; physical calibration remains pending.",
    }
    (OUT / "figure_source_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "rows": len(rows), "formats": ["svg", "pdf", "png", "tiff"],
                      "cost_table_mode": bool(cost_context)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
