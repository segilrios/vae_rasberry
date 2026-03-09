#!/usr/bin/env python3
"""
12_acq_sensor_to_zmq.py - Adaptador de adquisicion externa -> ZMQ IPC JSON PSD.

Objetivo:
- Integrar librerias/scripts de adquisicion SDR externos (ej. SDR-SpectrumMonitoring-Sensor)
- Publicar PSD en el endpoint IPC que consumen:
  - 11_edge_hackrf_psd_zmq_to_udp.py

Modos:
1) callable: importa un callable python (modulo:funcion).
2) script: ejecuta un script/proceso que emite JSON por stdout (1 JSON por linea).
"""

import argparse
import importlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

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


def extract_psd(frame: Any, preferred_key: str | None) -> np.ndarray | None:
    if isinstance(frame, dict):
        if preferred_key and preferred_key in frame:
            return _to_np1d(frame[preferred_key])
        for k in DEFAULT_KEYS:
            if k in frame:
                return _to_np1d(frame[k])
        return None
    return _to_np1d(frame)


def resolve_callable(target: str):
    if ":" not in target:
        raise ValueError("callable must be in format module:function")
    mod_name, fn_name = target.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise AttributeError(f"Function '{fn_name}' not found in module '{mod_name}'.")
    return fn


def add_repo_to_syspath(repo_path: str | None):
    if not repo_path:
        return
    p = Path(repo_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"--sensor_repo_path not found: {p}")
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def iter_from_callable(fn, kwargs: dict) -> Iterator[Any]:
    obj = fn(**kwargs)

    if isinstance(obj, Iterator):
        for item in obj:
            yield item
        return

    if isinstance(obj, Iterable) and not isinstance(obj, (dict, list, tuple, np.ndarray, str, bytes)):
        for item in obj:
            yield item
        return

    while True:
        item = fn(**kwargs)
        yield item


def iter_from_script(command: str) -> Iterator[Any]:
    proc = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # logs del proceso externo: se imprimen para debug y se ignoran
                print(f"[ACQ13] passthrough: {line}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["callable", "script"], required=True)
    ap.add_argument("--ipc", default="ipc:///tmp/ane_psd.ipc", help="PAIR endpoint where edge script is bound.")
    ap.add_argument("--out_key", default="psd_dbm", help="JSON key emitted to edge bridge.")
    ap.add_argument("--in_key", default=None, help="Preferred input key in source frames.")
    ap.add_argument("--sleep_ms", type=float, default=0.0, help="Throttle optional publisher delay.")
    ap.add_argument("--log_every", type=int, default=100)

    ap.add_argument("--sensor_repo_path", default=None, help="Path to external sensor repo to append to sys.path.")
    ap.add_argument("--callable", dest="callable_target", default=None, help="module:function")
    ap.add_argument("--callable_kwargs_json", default="{}", help="JSON dict kwargs for callable.")
    ap.add_argument("--script_cmd", default=None, help="Command producing JSON lines with PSD.")
    args = ap.parse_args()

    if args.mode == "callable" and not args.callable_target:
        raise ValueError("--callable is required for mode=callable")
    if args.mode == "script" and not args.script_cmd:
        raise ValueError("--script_cmd is required for mode=script")

    add_repo_to_syspath(args.sensor_repo_path)
    kwargs = json.loads(args.callable_kwargs_json or "{}")
    if not isinstance(kwargs, dict):
        raise ValueError("--callable_kwargs_json must be a JSON dict")

    if args.mode == "callable":
        fn = resolve_callable(args.callable_target)
        source_iter = iter_from_callable(fn, kwargs)
    else:
        source_iter = iter_from_script(args.script_cmd)

    ctx = zmq.Context.instance()
    zs = ctx.socket(zmq.PAIR)
    zs.connect(args.ipc)
    print(f"[ACQ13] mode={args.mode} -> connect {args.ipc} out_key={args.out_key}")

    n = 0
    t0 = time.perf_counter()

    try:
        for frame in source_iter:
            psd = extract_psd(frame, args.in_key)
            if psd is None:
                continue

            msg = {args.out_key: psd.tolist()}
            zs.send_string(json.dumps(msg))
            n += 1

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

            if args.log_every > 0 and (n % args.log_every == 0):
                dt = max(1e-9, time.perf_counter() - t0)
                print(f"[ACQ13] sent={n} fps={n/dt:.1f} bins={psd.shape[0]}")

    except KeyboardInterrupt:
        pass

    print("[ACQ13] stopped.")


if __name__ == "__main__":
    main()
