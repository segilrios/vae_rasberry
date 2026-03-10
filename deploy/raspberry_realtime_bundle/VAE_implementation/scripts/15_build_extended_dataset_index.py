#!/usr/bin/env python3
"""
15_build_extended_dataset_index.py - Consolida capturas extensas a un indice unico.

Escanea un directorio de adquisicion extendida y genera:
- merged_metadata.csv
- dataset_inventory.json

No transforma PSD. Solo construye el indice maestro para QA y preprocesamiento.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, help="Directorio con sesiones/capturas CSV.")
    ap.add_argument("--output_dir", required=True, help="Directorio de salida del indice consolidado.")
    args = ap.parse_args()

    root = repo_root()
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = root / input_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_root.rglob("*.csv"))
    csv_files = [p for p in csv_files if p.name not in ("acquisition_plan.csv", "merged_metadata.csv")]
    if not csv_files:
        raise FileNotFoundError(f"No capture CSV files found in {input_root}")

    parts = []
    session_rows = []
    total_frames = 0

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["source_file"] = str(csv_path.relative_to(input_root))
        if "row_index" not in df.columns:
            df["row_index"] = range(len(df))
        parts.append(df)
        total_frames += len(df)

        first = df.iloc[0].to_dict() if len(df) else {}
        session_rows.append(
            {
                "source_file": str(csv_path.relative_to(input_root)),
                "frames": int(len(df)),
                "session_id": first.get("session_id"),
                "sensor_id": first.get("sensor_id"),
                "location_label": first.get("location_label"),
                "scenario_label": first.get("scenario_label"),
                "profile_label": first.get("profile_label"),
                "center_freq_hz": first.get("center_freq_hz"),
                "sample_rate_hz": first.get("sample_rate_hz"),
                "rbw_hz": first.get("rbw_hz"),
                "lna_gain": first.get("lna_gain"),
                "vga_gain": first.get("vga_gain"),
                "window": first.get("window"),
                "overlap": first.get("overlap"),
                "decimation": first.get("decimation"),
            }
        )

    merged = pd.concat(parts, axis=0, ignore_index=True)
    merged_path = output_dir / "merged_metadata.csv"
    merged.to_csv(merged_path, index=False)

    inventory = {
        "input_root": str(input_root),
        "total_sessions": len(csv_files),
        "total_frames": int(total_frames),
        "sessions": session_rows,
    }
    inventory_path = output_dir / "dataset_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2))

    print("[INDEX15] merged_metadata:", merged_path)
    print("[INDEX15] inventory:", inventory_path)
    print("[INDEX15] sessions:", len(csv_files), "| frames:", total_frames)


if __name__ == "__main__":
    main()
