#!/usr/bin/env python3
"""
14_generate_acquisition_plan.py - Genera una matriz extendida de adquisicion.

Salida:
- acquisition_plan.csv
- acquisition_plan.jsonl
- profile_summary.json

Cada fila describe una sesion ejecutable con metadata y el cmd_json listo para
13_rf_engine_controller_to_zmq.py.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sanitize_label(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s).strip().lower())
    return out.strip("_") or "na"


def total_combinations(exp: dict, rf: dict, exe: dict) -> int:
    counts = [
        len(exp["location_labels"]),
        len(exp["scenario_labels"]),
        len(rf["center_freq_hz"]),
        len(rf["sample_rate_hz"]),
        len(rf["rbw_hz"]),
        len(rf["window"]),
        len(rf["overlap"]),
        len(rf["lna_gain"]),
        len(rf["vga_gain"]),
        len(rf["antenna_amp"]),
        len(rf["antenna_port"]),
        len(rf["decimation"]),
        len(exe["daily_time_blocks"]),
        int(exp["repeats_per_profile"]),
    ]
    return math.prod(counts)


def iter_rf_profiles(rf: dict):
    for combo in itertools.product(
        rf["center_freq_hz"],
        rf["sample_rate_hz"],
        rf["rbw_hz"],
        rf["window"],
        rf["overlap"],
        rf["lna_gain"],
        rf["vga_gain"],
        rf["antenna_amp"],
        rf["antenna_port"],
        rf["decimation"],
    ):
        yield {
            "center_freq_hz": int(combo[0]),
            "sample_rate_hz": int(combo[1]),
            "rbw_hz": int(combo[2]),
            "window": str(combo[3]),
            "overlap": float(combo[4]),
            "lna_gain": int(combo[5]),
            "vga_gain": int(combo[6]),
            "antenna_amp": bool(combo[7]),
            "antenna_port": int(combo[8]),
            "decimation": int(combo[9]),
        }


def iter_contexts(exp: dict, exe: dict):
    for combo in itertools.product(
        exp["location_labels"],
        exp["scenario_labels"],
        exe["daily_time_blocks"],
        range(int(exp["repeats_per_profile"])),
    ):
        yield {
            "location_label": combo[0],
            "scenario_label": combo[1],
            "time_block": combo[2],
            "repeat_idx": int(combo[3]),
        }


def build_balanced_pairs(profiles: list, contexts: list, limit: int):
    total_possible = len(profiles) * len(contexts)
    target = min(limit, total_possible)
    written = 0
    for profile_idx in range(len(profiles)):
        context_offset = profile_idx % len(contexts)
        for step in range(len(contexts)):
            ctx_idx = (context_offset + step) % len(contexts)
            yield profiles[profile_idx], contexts[ctx_idx]
            written += 1
            if written >= target:
                return


def resolve_target_sessions(exp: dict, total_possible: int, cli_max_sessions: int) -> int:
    planning = exp.get("planning", {})
    if cli_max_sessions > 0:
        target = int(cli_max_sessions)
    elif int(planning.get("target_sessions", 0)) > 0:
        target = int(planning["target_sessions"])
    elif float(planning.get("target_hours", 0)) > 0:
        target = int(float(planning["target_hours"]) * 3600.0 / int(exp["capture_duration_s"]))
    else:
        target = total_possible

    hard_cap = int(planning.get("max_sessions_hard_cap", 0))
    if hard_cap > 0:
        target = min(target, hard_cap)
    return min(max(1, target), total_possible)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_sessions", type=int, default=0, help="0=all. Limit output rows for planning.")
    ap.add_argument("--print_only", action="store_true", help="Only print size estimate and exit.")
    ap.add_argument("--progress_every", type=int, default=10000)
    args = ap.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    exp = cfg["experiment"]
    rf = cfg["rf"]
    exe = cfg["execution"]

    exp_name = sanitize_label(exp["name"])
    out_root = root / exp["output_root"] / exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    combos_total = total_combinations(exp, rf, exe)
    profiles = list(iter_rf_profiles(rf))
    contexts = list(iter_contexts(exp, exe))
    sessions_to_write = resolve_target_sessions(exp, combos_total, int(args.max_sessions))

    print("[PLAN14] estimated sessions:", combos_total)
    print("[PLAN14] estimated hours:", round(combos_total * int(exp["capture_duration_s"]) / 3600.0, 2))
    print("[PLAN14] rf_profiles:", len(profiles), "| contexts:", len(contexts))
    print("[PLAN14] planned sessions:", sessions_to_write)
    print("[PLAN14] planned hours:", round(sessions_to_write * int(exp["capture_duration_s"]) / 3600.0, 2))
    if args.print_only:
        return

    fieldnames = [
        "session_id",
        "experiment_name",
        "sensor_id",
        "location_label",
        "scenario_label",
        "time_block",
        "repeat_idx",
        "capture_duration_s",
        "profile_label",
        "decimation",
        "capture_relpath",
        "cmd_json",
        "notes",
    ]

    csv_path = out_root / "acquisition_plan.csv"
    jsonl_path = out_root / "acquisition_plan.jsonl"

    csv_fh = csv_path.open("w", newline="", encoding="utf-8")
    jsonl_fh = jsonl_path.open("w", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    writer.writeheader()

    written = 0
    for i, (profile, context) in enumerate(build_balanced_pairs(profiles, contexts, sessions_to_write), start=1):
        location_label = context["location_label"]
        scenario_label = context["scenario_label"]
        time_block = context["time_block"]
        repeat_idx = context["repeat_idx"]
        center_freq_hz = profile["center_freq_hz"]
        sample_rate_hz = profile["sample_rate_hz"]
        rbw_hz = profile["rbw_hz"]
        window = profile["window"]
        overlap = profile["overlap"]
        lna_gain = profile["lna_gain"]
        vga_gain = profile["vga_gain"]
        antenna_amp = profile["antenna_amp"]
        antenna_port = profile["antenna_port"]
        decimation = profile["decimation"]

        profile_label = (
            f"cf_{center_freq_hz}_sr_{sample_rate_hz}_rbw_{rbw_hz}_"
            f"win_{window}_ov_{overlap}_lna_{lna_gain}_vga_{vga_gain}_"
            f"amp_{int(bool(antenna_amp))}_dec_{decimation}"
        )
        session_id = f"{i:06d}_{sanitize_label(location_label)}_{sanitize_label(scenario_label)}_r{repeat_idx+1}"
        capture_rel = f"{exp_name}/{session_id}/capture.csv"
        cmd = {
            "center_freq_hz": center_freq_hz,
            "sample_rate_hz": sample_rate_hz,
            "rbw_hz": rbw_hz,
            "window": window,
            "overlap": overlap,
            "lna_gain": lna_gain,
            "vga_gain": vga_gain,
            "antenna_amp": antenna_amp,
            "antenna_port": antenna_port,
        }

        row = {
            "session_id": session_id,
            "experiment_name": exp_name,
            "sensor_id": exp["sensor_id"],
            "location_label": location_label,
            "scenario_label": scenario_label,
            "time_block": time_block,
            "repeat_idx": int(repeat_idx),
            "capture_duration_s": int(exp["capture_duration_s"]),
            "profile_label": profile_label,
            "decimation": int(decimation),
            "capture_relpath": capture_rel,
            "cmd_json": json.dumps(cmd, separators=(",", ":")),
            "notes": exp.get("notes", ""),
        }
        writer.writerow(row)
        jsonl_fh.write(json.dumps(row) + "\n")
        written += 1

        if args.progress_every > 0 and written % int(args.progress_every) == 0:
            print(f"[PLAN14] written {written}/{sessions_to_write}")

    csv_fh.close()
    jsonl_fh.close()

    summary = {
        "experiment_name": exp_name,
        "total_sessions": int(written),
        "total_sessions_possible": int(combos_total),
        "total_locations": len(exp["location_labels"]),
        "total_scenarios": len(exp["scenario_labels"]),
        "repeats_per_profile": int(exp["repeats_per_profile"]),
        "capture_duration_s": int(exp["capture_duration_s"]),
        "estimated_total_hours": round(written * int(exp["capture_duration_s"]) / 3600.0, 2),
    }
    (out_root / "profile_summary.json").write_text(json.dumps(summary, indent=2))

    print("[PLAN14] output:", out_root)
    print("[PLAN14] sessions_written:", written)
    print("[PLAN14] csv:", csv_path)
    print("[PLAN14] jsonl:", jsonl_path)


if __name__ == "__main__":
    main()
