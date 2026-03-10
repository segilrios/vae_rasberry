#!/usr/bin/env python3
"""
16_run_acquisition_session.py - Ejecuta una sesion del acquisition_plan.csv.

Lee una fila del plan generado por 14_generate_acquisition_plan.py y ejecuta
13_rf_engine_controller_to_zmq.py con todos los parametros necesarios, durante
el tiempo de captura indicado en la fila.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_rows(plan_csv: Path) -> list[dict]:
    with plan_csv.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def select_row(rows: list[dict], session_id: str | None, row_index: int | None) -> dict:
    if session_id:
        for row in rows:
            if row["session_id"] == session_id:
                return row
        raise ValueError(f"session_id not found: {session_id}")
    if row_index is None:
        if not rows:
            raise ValueError("plan has no rows")
        return rows[0]
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index out of range: {row_index}")
    return rows[row_index]


def build_command(args, row: dict, root: Path) -> tuple[list[str], Path, int]:
    controller_script = Path(args.controller_script)
    if not controller_script.is_absolute():
        controller_script = root / controller_script

    base_capture_dir = Path(args.base_capture_dir)
    if not base_capture_dir.is_absolute():
        base_capture_dir = root / base_capture_dir
    capture_csv = base_capture_dir / row["capture_relpath"]

    duration_s = int(args.duration_override_s) if args.duration_override_s > 0 else int(row["capture_duration_s"])

    cmd = [
        args.python_bin,
        str(controller_script),
        "--rf_ipc",
        args.rf_ipc,
        "--out_ipc",
        args.out_ipc,
        "--in_key",
        args.in_key,
        "--out_key",
        args.out_key,
        "--capture_csv",
        str(capture_csv),
        "--session_id",
        row["session_id"],
        "--sensor_id",
        row["sensor_id"],
        "--location_label",
        row["location_label"],
        "--scenario_label",
        row["scenario_label"],
        "--profile_label",
        row["profile_label"],
        "--decimation",
        str(row["decimation"]),
        "--cmd_json",
        row["cmd_json"],
        "--send_cmd_every_s",
        str(args.send_cmd_every_s),
        "--log_every",
        str(args.log_every),
        "--notes",
        row.get("notes", ""),
    ]
    return cmd, capture_csv, duration_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_csv", required=True)
    ap.add_argument("--session_id", default=None)
    ap.add_argument("--row_index", type=int, default=None)
    ap.add_argument("--base_capture_dir", default="data/raw/extended_acquisition")
    ap.add_argument("--controller_script", default="VAE_implementation/scripts/13_rf_engine_controller_to_zmq.py")
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--rf_ipc", default="ipc:///tmp/rf_engine")
    ap.add_argument("--out_ipc", default="ipc:///tmp/ane_psd.ipc")
    ap.add_argument("--in_key", default="Pxx")
    ap.add_argument("--out_key", default="psd_dbm")
    ap.add_argument("--send_cmd_every_s", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--duration_override_s", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    plan_csv = Path(args.plan_csv)
    if not plan_csv.is_absolute():
        plan_csv = root / plan_csv
    if not plan_csv.exists():
        raise FileNotFoundError(f"plan_csv not found: {plan_csv}")

    rows = load_rows(plan_csv)
    row = select_row(rows, args.session_id, args.row_index)
    cmd, capture_csv, duration_s = build_command(args, row, root)

    print("[RUN16] session_id:", row["session_id"])
    print("[RUN16] capture_csv:", capture_csv)
    print("[RUN16] duration_s:", duration_s)
    print("[RUN16] cmd:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))

    if args.dry_run:
        return

    capture_csv.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd)
    start = time.time()
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"controller exited early with code {proc.returncode}")
            if (time.time() - start) >= duration_s:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("[RUN16] completed session:", row["session_id"])


if __name__ == "__main__":
    main()
