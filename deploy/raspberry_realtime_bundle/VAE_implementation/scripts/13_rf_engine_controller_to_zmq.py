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

    print("[CTRL13] stopped.")


if __name__ == "__main__":
    main()
