#!/usr/bin/env python3
"""
13_rf_engine_controller_to_zmq.py - Controller puente para rf_engine -> IPC PSD.

Uso principal:
- No modificar SDR-SpectrumMonitoring-Sensor.
- Levantar rf_engine standalone (sin orchestrator).
- Este script hace de peer ZMQ PAIR del rf_engine, envia comando(s) y re-publica la PSD
  al IPC consumido por 11_edge_hackrf_psd_zmq_to_udp.py.

Topologia:
  rf_engine (connect PAIR -> ipc:///tmp/rf_engine)
        <-> this controller (bind ipc:///tmp/rf_engine)
  this controller (connect ipc:///tmp/ane_psd.ipc)
        -> 11_edge_hackrf_psd_zmq_to_udp.py (bind ipc:///tmp/ane_psd.ipc)
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq


DEFAULT_KEYS = ("psd_dbm", "psd", "p_out", "power_dbm", "p_dbm", "spectrum_dbm", "bins_dbm", "pxx")


def _to_np1d(x: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def pick_psd(obj: dict, preferred_key: str | None) -> np.ndarray | None:
    if preferred_key and preferred_key in obj:
        return _to_np1d(obj[preferred_key])
    for k in DEFAULT_KEYS:
        if k in obj:
            return _to_np1d(obj[k])
    return None


def pick_scalar(obj: dict, key: str, default):
    val = obj.get(key, default)
    return default if val is None else val


def open_capture_csv(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fh = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "session_id",
            "sensor_id",
            "location_label",
            "scenario_label",
            "profile_label",
            "timestamp",
            "created_at",
            "center_freq_hz",
            "sample_rate_hz",
            "rbw_hz",
            "window",
            "overlap",
            "lna_gain",
            "vga_gain",
            "antenna_amp",
            "antenna_port",
            "decimation",
            "start_freq_hz",
            "end_freq_hz",
            "pxx",
            "notes",
        ],
    )
    if not exists:
        writer.writeheader()
    return fh, writer


def load_command_payload(cmd_json: str, cmd_file: str | None) -> dict:
    if cmd_file:
        p = Path(cmd_file)
        if not p.exists():
            raise FileNotFoundError(f"--cmd_file not found: {p}")
        data = json.loads(p.read_text())
    else:
        data = json.loads(cmd_json)
    if not isinstance(data, dict):
        raise ValueError("Command payload must be a JSON object.")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_ipc", default="ipc:///tmp/rf_engine", help="PAIR endpoint where rf_engine connects.")
    ap.add_argument("--out_ipc", default="ipc:///tmp/ane_psd.ipc", help="PAIR endpoint where 11_edge is bound.")
    ap.add_argument("--in_key", default=None, help="Preferred key for PSD inside rf_engine payload.")
    ap.add_argument("--out_key", default="psd_dbm", help="JSON key sent to 11_edge.")

    ap.add_argument("--cmd_json", default="{}", help="JSON command sent to rf_engine.")
    ap.add_argument("--cmd_file", default=None, help="Path to JSON file with command payload.")
    ap.add_argument("--send_cmd_every_s", type=float, default=1.0, help="0=send once at start.")
    ap.add_argument("--rf_rcv_timeout_ms", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--capture_csv", default=None, help="Optional CSV output compatible with 01_preprocess.py.")
    ap.add_argument("--session_id", default="session_current")
    ap.add_argument("--sensor_id", default="sensor_01")
    ap.add_argument("--location_label", default="unknown")
    ap.add_argument("--scenario_label", default="unspecified")
    ap.add_argument("--profile_label", default="default_profile")
    ap.add_argument("--decimation", type=int, default=1)
    ap.add_argument("--notes", default="")
    ap.add_argument("--flush_every", type=int, default=20)
    args = ap.parse_args()

    cmd = load_command_payload(args.cmd_json, args.cmd_file)

    ctx = zmq.Context.instance()

    rf_sock = ctx.socket(zmq.PAIR)
    rf_sock.setsockopt(zmq.LINGER, 0)
    rf_sock.setsockopt(zmq.RCVTIMEO, int(args.rf_rcv_timeout_ms))
    rf_sock.bind(args.rf_ipc)

    out_sock = ctx.socket(zmq.PAIR)
    out_sock.setsockopt(zmq.LINGER, 0)
    out_sock.connect(args.out_ipc)

    print(f"[CTRL13] bind rf_ipc={args.rf_ipc}")
    print(f"[CTRL13] connect out_ipc={args.out_ipc}")

    cmd_msg = json.dumps(cmd, separators=(",", ":"))
    rf_sock.send_string(cmd_msg)
    print("[CTRL13] initial command sent")
    last_cmd_t = time.perf_counter()

    capture_fh = None
    capture_writer = None
    if args.capture_csv:
        capture_fh, capture_writer = open_capture_csv(Path(args.capture_csv))
        print(f"[CTRL13] capture_csv={args.capture_csv}")

    rx = 0
    fw = 0
    t0 = time.perf_counter()

    try:
        while True:
            if args.send_cmd_every_s > 0 and (time.perf_counter() - last_cmd_t) >= args.send_cmd_every_s:
                rf_sock.send_string(cmd_msg)
                last_cmd_t = time.perf_counter()

            try:
                msg = rf_sock.recv_string()
            except zmq.error.Again:
                continue

            rx += 1
            try:
                obj = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict):
                continue

            psd = pick_psd(obj, args.in_key)
            if psd is None:
                continue

            if capture_writer is not None:
                row = {
                    "session_id": args.session_id,
                    "sensor_id": args.sensor_id,
                    "location_label": args.location_label,
                    "scenario_label": args.scenario_label,
                    "profile_label": args.profile_label,
                    "timestamp": pick_scalar(obj, "timestamp", int(time.time() * 1000)),
                    "created_at": int(time.time() * 1000),
                    "center_freq_hz": pick_scalar(obj, "center_freq_hz", cmd.get("center_freq_hz")),
                    "sample_rate_hz": pick_scalar(obj, "sample_rate_hz", cmd.get("sample_rate_hz")),
                    "rbw_hz": pick_scalar(obj, "rbw_hz", cmd.get("rbw_hz")),
                    "window": pick_scalar(obj, "window", cmd.get("window")),
                    "overlap": pick_scalar(obj, "overlap", cmd.get("overlap")),
                    "lna_gain": pick_scalar(obj, "lna_gain", cmd.get("lna_gain")),
                    "vga_gain": pick_scalar(obj, "vga_gain", cmd.get("vga_gain")),
                    "antenna_amp": pick_scalar(obj, "antenna_amp", cmd.get("antenna_amp")),
                    "antenna_port": pick_scalar(obj, "antenna_port", cmd.get("antenna_port")),
                    "decimation": args.decimation,
                    "start_freq_hz": pick_scalar(obj, "start_freq_hz", None),
                    "end_freq_hz": pick_scalar(obj, "end_freq_hz", None),
                    "pxx": json.dumps(psd.tolist(), separators=(",", ":")),
                    "notes": args.notes,
                }
                capture_writer.writerow(row)
                if args.flush_every > 0 and (fw + 1) % args.flush_every == 0:
                    capture_fh.flush()

            out_obj = {args.out_key: psd.tolist()}
            out_sock.send_string(json.dumps(out_obj))
            fw += 1

            if args.log_every > 0 and (fw % args.log_every == 0):
                dt = max(1e-9, time.perf_counter() - t0)
                print(f"[CTRL13] rx={rx} fw={fw} fw_fps={fw/dt:.1f} bins={psd.shape[0]}")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            rf_sock.close(0)
        except Exception:
            pass
        try:
            out_sock.close(0)
        except Exception:
            pass
        if capture_fh is not None:
            capture_fh.flush()
            capture_fh.close()

    print("[CTRL13] stopped.")


if __name__ == "__main__":
    main()
