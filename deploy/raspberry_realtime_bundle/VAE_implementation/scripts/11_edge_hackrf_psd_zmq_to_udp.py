#!/usr/bin/env python3
"""
11_edge_hackrf_psd_zmq_to_udp.py - Edge bridge (production):
ZMQ JSON PSD -> normalize/resample -> encoder TFLite INT8 -> UDP packet stream.

Notes:
- This script does not compute PSD; it consumes PSD produced by your RF pipeline.
- Uses the same packet format as 10_udp_sender_prod.py (KLP1 + zlib block).
"""

import argparse
import importlib.util
import json
import socket
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ModuleNotFoundError:
    import tensorflow as tf

    TFLiteInterpreter = tf.lite.Interpreter


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_pack_module() -> object:
    root = repo_root()
    mod_path = root / "VAE_implementation" / "scripts" / "07_pack_unpack.py"
    spec = importlib.util.spec_from_file_location("packmod", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def load_npz_dict(processed_dir: Path) -> dict:
    npz_path = processed_dir / "dataset_psd_1024_norm.npz"
    if not npz_path.exists():
        candidates = sorted(processed_dir.glob("*.npz"), key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No .npz found in {processed_dir}")
        npz_path = candidates[0]
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_tflite(tflite_path: Path):
    interp = TFLiteInterpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    return interp, in_det, out_det


def infer_mu_int8_one(interp, in_det, out_det, x_norm_1024: np.ndarray) -> np.ndarray:
    in_scale, in_zp = in_det["quantization"]
    x = x_norm_1024[None, :, None].astype(np.float32)

    if in_det["dtype"] == np.int8:
        xq = np.round(x / in_scale + in_zp).astype(np.int8)
    elif in_det["dtype"] == np.uint8:
        xq = np.round(x / in_scale + in_zp).astype(np.uint8)
    else:
        xq = x.astype(in_det["dtype"])

    interp.set_tensor(in_det["index"], xq)
    interp.invoke()
    yq = interp.get_tensor(out_det["index"])[0]
    return yq.astype(np.int8, copy=False)


def resample_to_1024(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 1024:
        return x.astype(np.float32, copy=False)
    xp = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=True)
    xq = np.linspace(0.0, 1.0, num=1024, endpoint=True)
    return np.interp(xq, xp, x).astype(np.float32)


def normalize_global_minmax(x: np.ndarray, gmin: float, gmax: float) -> np.ndarray:
    xc = np.clip(x, gmin, gmax)
    return (xc - gmin) / (gmax - gmin + 1e-8)


def guess_psd_array(obj: dict, preferred_key: str | None) -> np.ndarray | None:
    if preferred_key and preferred_key in obj:
        return np.asarray(obj[preferred_key], dtype=np.float32)
    for k in ("psd_dbm", "psd", "p_out", "power_dbm", "p_dbm", "spectrum_dbm", "bins_dbm", "pxx"):
        if k in obj:
            return np.asarray(obj[k], dtype=np.float32)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_global_best", action="store_true")
    ap.add_argument("--run_name", default=None)

    ap.add_argument("--ipc", default="ipc:///tmp/ane_psd.ipc")
    ap.add_argument("--psd_key", default=None)

    ap.add_argument("--dest_ip", required=True)
    ap.add_argument("--port", type=int, default=5005)

    ap.add_argument("--block_len", type=int, default=30)
    ap.add_argument("--zlib_level", type=int, default=1)
    ap.add_argument("--log_every_packets", type=int, default=10)
    ap.add_argument("--already_normalized", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    processed_dir = root / cfg["paths"]["processed_dir"]
    models_dir = root / cfg["paths"]["models_dir"]

    if args.use_global_best:
        model_dir = models_dir / "GLOBAL_BEST"
        tag = "GLOBAL_BEST"
    else:
        run = args.run_name or cfg.get("train", {}).get("run_name", "run_current")
        model_dir = models_dir / "runs" / run
        tag = f"RUN:{run}"

    tflite_path = model_dir / "encoder_mu_int8.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(f"Missing TFLite encoder: {tflite_path}")

    packmod = load_pack_module()
    interp, in_det, out_det = load_tflite(tflite_path)

    gmin = gmax = None
    if not args.already_normalized:
        npz = load_npz_dict(processed_dir)
        mode = str(npz.get("normalize_mode", "global_minmax"))
        gmin = float(npz.get("gmin", np.nan))
        gmax = float(npz.get("gmax", np.nan))
        if mode != "global_minmax" or not np.isfinite(gmin + gmax):
            raise ValueError("Need global_minmax metadata in npz or use --already_normalized")

    L = int(args.block_len)
    if L <= 0 or L > 255:
        raise ValueError("block_len must be in [1,255]")

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.dest_ip, args.port)

    ctx = zmq.Context.instance()
    zs = ctx.socket(zmq.PAIR)
    zs.bind(args.ipc)
    zs.RCVTIMEO = 1000

    print(f"[EDGE11] ZMQ bind {args.ipc} | UDP -> {args.dest_ip}:{args.port} | {tag}")

    seq = 0
    frames = 0
    bytes_total = 0
    t0 = time.perf_counter()

    mu_block = np.zeros((L, 32), dtype=np.int8)
    bi = 0

    while True:
        try:
            msg = zs.recv_string()
        except zmq.error.Again:
            continue
        except KeyboardInterrupt:
            break

        try:
            obj = json.loads(msg)
        except json.JSONDecodeError:
            continue

        psd = guess_psd_array(obj, args.psd_key)
        if psd is None:
            continue

        x = np.asarray(psd, dtype=np.float32).reshape(-1)
        x = resample_to_1024(x)
        x_norm = x if args.already_normalized else normalize_global_minmax(x, gmin, gmax)

        mu_block[bi] = infer_mu_int8_one(interp, in_det, out_det, x_norm)
        bi += 1
        frames += 1

        if bi >= L:
            pkt = packmod.pack_packet(mu_block, seq=seq, zlib_level=args.zlib_level, keyframe=True)
            udp.sendto(pkt, dest)
            bytes_total += len(pkt)
            seq += 1
            bi = 0

            if args.log_every_packets > 0 and (seq % args.log_every_packets == 0):
                dt = max(1e-9, time.perf_counter() - t0)
                print(
                    f"[EDGE11] pkts={seq} frames={frames} avgB/f={bytes_total/max(1,frames):.2f} "
                    f"frames/s={frames/dt:.1f} KB/s={bytes_total/dt/1024:.2f}"
                )

    print("[EDGE11] stopped.")


if __name__ == "__main__":
    main()
