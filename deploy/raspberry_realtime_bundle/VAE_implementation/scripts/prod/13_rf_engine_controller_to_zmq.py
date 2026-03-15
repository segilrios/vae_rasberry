#!/usr/bin/env python3
"""Bridge ``rf_engine`` output into the repository IPC PSD stream.

Primary use case:
- keep ``SDR-SpectrumMonitoring-Sensor`` unchanged,
- run ``rf_engine`` in standalone mode without its orchestrator,
- act as the ZMQ ``PAIR`` peer for ``rf_engine``, send control command(s), and
  republish the resulting PSD frames to the IPC endpoint consumed by
  ``11_edge_hackrf_psd_zmq_to_udp.py``.

Topology:
  rf_engine (connect PAIR -> ipc:///tmp/rf_engine)
        <-> this controller (bind ipc:///tmp/rf_engine)
  this controller (connect ipc:///tmp/ane_psd.ipc)
        -> 11_edge_hackrf_psd_zmq_to_udp.py (bind ipc:///tmp/ane_psd.ipc)
"""

import argparse
import json
import time
from typing import Any

import numpy as np
import zmq


DEFAULT_KEYS = (
    "psd_dbm",
    "psd",
    "p_out",
    "power_dbm",
    "p_dbm",
    "spectrum_dbm",
    "bins_dbm",
    "pxx",
    "Pxx",
)


def _to_np1d(x: Any) -> np.ndarray | None:
    """Convert one PSD-like payload into a flat ``float32`` NumPy array."""

    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def pick_psd(obj: dict, preferred_key: str | None) -> np.ndarray | None:
    """Extract the PSD vector from one decoded payload mapping."""

    if preferred_key and preferred_key in obj:
        return _to_np1d(obj[preferred_key])
    for k in DEFAULT_KEYS:
        if k in obj:
            return _to_np1d(obj[k])
    return None


def main():
    """Parse CLI arguments and relay ``rf_engine`` PSD messages to the edge IPC."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_ipc", default="ipc:///tmp/rf_engine")
    ap.add_argument("--out_ipc", default="ipc:///tmp/ane_psd.ipc")
    ap.add_argument(
        "--in_key", default=None, help="Preferred key in rf_engine payload (e.g., Pxx)."
    )
    ap.add_argument("--out_key", default="psd_dbm", help="Key emitted to edge bridge.")
    ap.add_argument(
        "--cmd_json", default=None, help="JSON string command sent to rf_engine."
    )
    ap.add_argument("--cmd_file", default=None, help="Path to JSON file with command.")
    ap.add_argument("--send_cmd_every_s", type=float, default=1.0)
    ap.add_argument("--rf_rcv_timeout_ms", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    if args.cmd_json and args.cmd_file:
        raise ValueError("Use only one of --cmd_json or --cmd_file")

    if args.cmd_file:
        with open(args.cmd_file, "r", encoding="utf-8") as fh:
            cmd = json.load(fh)
    elif args.cmd_json:
        cmd = json.loads(args.cmd_json)
    else:
        cmd = {}

    ctx = zmq.Context.instance()
    rf_sock = ctx.socket(zmq.PAIR)
    rf_sock.bind(args.rf_ipc)
    rf_sock.RCVTIMEO = int(args.rf_rcv_timeout_ms)
    print(f"[CTRL13] bind rf_ipc={args.rf_ipc}")

    out_sock = ctx.socket(zmq.PAIR)
    out_sock.connect(args.out_ipc)
    print(f"[CTRL13] connect out_ipc={args.out_ipc}")

    n_rx = 0
    n_fw = 0
    t0 = time.perf_counter()
    t_cmd = 0.0

    try:
        if cmd:
            rf_sock.send_string(json.dumps(cmd))
            print("[CTRL13] initial command sent")

        while True:
            now = time.perf_counter()
            if (
                args.send_cmd_every_s > 0
                and (now - t_cmd) >= args.send_cmd_every_s
                and cmd
            ):
                rf_sock.send_string(json.dumps(cmd))
                t_cmd = now

            try:
                msg = rf_sock.recv_string()
            except zmq.error.Again:
                continue

            try:
                obj = json.loads(msg)
            except json.JSONDecodeError:
                continue

            n_rx += 1
            psd = pick_psd(obj, args.in_key)
            if psd is None:
                continue

            out_sock.send_string(json.dumps({args.out_key: psd.tolist()}))
            n_fw += 1

            if args.log_every > 0 and (n_fw % args.log_every == 0):
                dt = max(1e-9, time.perf_counter() - t0)
                print(
                    f"[CTRL13] rx={n_rx} fw={n_fw} fw_fps={n_fw / dt:.1f} bins={psd.shape[0]}"
                )

    except KeyboardInterrupt:
        pass

    print("[CTRL13] stopped.")


if __name__ == "__main__":
    main()
