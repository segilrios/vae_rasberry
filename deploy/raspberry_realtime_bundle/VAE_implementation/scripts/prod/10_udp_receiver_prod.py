#!/usr/bin/env python3
"""
10_udp_receiver_prod.py ??? Production UDP receiver (server side).

Receives production packets (no indices):
- packet -> decode_packet_to_mu() -> mu_int8 block (L,32)
- dequant mu using encoder_mu_int8.tflite output quant params
- reconstruct x_norm using decoder weights
- optionally invert normalization back to original scale (global_minmax via gmin/gmax in npz)
- save plots (waterfall + last PSD line)

Robust:
- no crash on timeouts
- idle_stop optional
- seq gap / out-of-order logging

Usage:
  python VAE_implementation/scripts/prod/10_udp_receiver_prod.py --config ... --use_global_best --bind_ip 0.0.0.0 --port 5005 --idle_stop_s 10

Outputs:
  <model_dir>/udp_prod/
"""

import argparse
import importlib.util
import socket
import time
from pathlib import Path
import json

import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from importlib.machinery import ModuleSpec
from tensorflow import keras  # type: ignore[import-untyped]
from tensorflow.keras import layers  # type: ignore[import-untyped]

import tensorflow as tf  # type: ignore[import-untyped]

try:
    import psutil  # type: ignore[import-not-found]
except ModuleNotFoundError:
    psutil = None


def repo_root() -> Path:
    """Return the project root for repository-relative config paths."""

    return Path(__file__).resolve().parents[3]


def _resolve_tflite_interpreter() -> type:
    """Load the preferred TFLite interpreter class on demand."""

    try:
        from tflite_runtime.interpreter import (  # type: ignore[import-not-found]
            Interpreter as interpreter_cls,
        )

        return interpreter_cls
    except ModuleNotFoundError:
        return tf.lite.Interpreter


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_pack_module() -> object:
    root = repo_root()
    mod_path = root / "VAE_implementation" / "scripts" / "codec" / "07_pack_unpack.py"
    spec = importlib.util.spec_from_file_location("packmod", str(mod_path))
    if spec is None or not isinstance(spec.loader, object):
        raise ImportError(f"Could not load packet codec module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    if not isinstance(spec, ModuleSpec) or spec.loader is None:
        raise ImportError(f"Invalid import spec for packet codec module: {mod_path}")
    spec.loader.exec_module(mod)
    return mod


def load_train_module() -> object:
    root = repo_root()
    mod_path = root / "VAE_implementation" / "scripts" / "training" / "02_train.py"
    spec = importlib.util.spec_from_file_location("trainmod", str(mod_path))
    if spec is None or not isinstance(spec.loader, object):
        raise ImportError(f"Could not load train module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    if not isinstance(spec, ModuleSpec) or spec.loader is None:
        raise ImportError(f"Invalid import spec for train module: {mod_path}")
    spec.loader.exec_module(mod)
    return mod


def load_npz(processed_dir: Path) -> dict:
    npz_path = processed_dir / "dataset_psd_1024_norm.npz"
    if not npz_path.exists():
        candidates = sorted(
            processed_dir.glob("*.npz"), key=lambda p: p.stat().st_size, reverse=True
        )
        if not candidates:
            raise FileNotFoundError(f"No .npz found in {processed_dir}")
        npz_path = candidates[0]
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_splits(processed_dir: Path):
    sdir = processed_dir / "splits"
    tr = np.load(sdir / "train_idx.npy")
    va = np.load(sdir / "val_idx.npy")
    te = np.load(sdir / "test_idx.npy")
    return tr, va, te


def invert_global_minmax(x_norm: np.ndarray, gmin: float, gmax: float) -> np.ndarray:
    return x_norm * (gmax - gmin) + gmin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_global_best", action="store_true")
    ap.add_argument("--run_name", default=None)

    ap.add_argument("--bind_ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5005)

    ap.add_argument("--socket_timeout_s", type=float, default=1.0)
    ap.add_argument(
        "--idle_stop_s",
        type=float,
        default=0.0,
        help="Stop if no packets for X seconds (0=never).",
    )
    ap.add_argument(
        "--max_packets", type=int, default=0, help="0=run until idle_stop or Ctrl+C"
    )

    ap.add_argument("--waterfall_max_frames", type=int, default=300)
    ap.add_argument(
        "--save_every_packets", type=int, default=10, help="Save plots every N packets."
    )
    ap.add_argument(
        "--invert_norm_to_original",
        action="store_true",
        help="Use gmin/gmax from npz to return to original scale.",
    )
    ap.add_argument(
        "--compare_split",
        choices=["train", "val", "test"],
        default=None,
        help="Compare against original frames from this split.",
    )
    ap.add_argument(
        "--compare_realtime",
        action="store_true",
        help="Compare against original PSDs carried in packets (edge --send_orig).",
    )
    ap.add_argument(
        "--plot_every_packets",
        type=int,
        default=1,
        help="Save overlay every N packets in compare mode.",
    )
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

    out_dir = model_dir / "udp_prod"
    ensure_dir(out_dir)

    packmod = load_pack_module()

    # Load decoder (use architecture from config)
    dec_w = model_dir / "dec_best.weights.h5"
    if not dec_w.exists():
        raise FileNotFoundError(f"Decoder weights not found: {dec_w}")
    train_cfg = cfg.get("train", {})
    enc_strides = tuple(train_cfg.get("enc_strides", [2, 2]))
    dec_channels = tuple(train_cfg.get("dec_channels", [32, 16]))
    dec_kernels = tuple(train_cfg.get("dec_kernels", [3, 5]))
    dec_strides = tuple(train_cfg.get("dec_strides", [2, 2]))
    padding = str(train_cfg.get("padding", "same"))
    trainmod = load_train_module()
    decoder = trainmod.build_decoder(
        input_bins=int(cfg.get("preprocess", {}).get("target_bins", 1024)),
        latent_dim=32,
        dec_channels=dec_channels,
        dec_kernels=dec_kernels,
        dec_strides=dec_strides,
        enc_strides=enc_strides,
        padding=padding,
    )
    _ = decoder(tf.zeros((1, 32), dtype=tf.float32), training=False)
    decoder.load_weights(dec_w)

    # Load encoder tflite only for output quant params
    tflite_path = model_dir / "encoder_mu_int8.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite encoder not found: {tflite_path}")
    tfl = _resolve_tflite_interpreter()(model_path=str(tflite_path))
    tfl.allocate_tensors()
    out_det = tfl.get_output_details()[0]
    out_scale, out_zp = out_det["quantization"]
    print("[RECV10] Dequant:", "scale=", out_scale, "zp=", out_zp)

    # Optional inverse normalization params
    npz = load_npz(processed_dir)
    norm_mode = str(npz.get("normalize_mode", "global_minmax"))
    gmin = float(npz.get("gmin", np.nan))
    gmax = float(npz.get("gmax", np.nan))
    if args.invert_norm_to_original:
        if norm_mode != "global_minmax" or not np.isfinite(gmin + gmax):
            print(
                "[RECV10] WARN invert_norm_to_original requested but npz lacks valid global_minmax params. Will save normalized scale."
            )
            args.invert_norm_to_original = False
        else:
            print("[RECV10] Inverting normalization to original scale using gmin/gmax.")

    compare_enabled = args.compare_split is not None or args.compare_realtime
    X_ref = None
    ref_cursor = 0
    if compare_enabled and args.compare_split is not None:
        tr, va, te = load_splits(processed_dir)
        split_idx = {"train": tr, "val": va, "test": te}[args.compare_split].astype(
            np.int64
        )
        X_ref = npz["X"].astype(np.float32)[split_idx]
        if args.invert_norm_to_original and norm_mode == "global_minmax":
            X_ref = invert_global_minmax(X_ref, gmin, gmax)
        if norm_mode == "per_frame_minmax":
            meta_path = processed_dir / "metadata.csv"
            if meta_path.exists():
                import pandas as pd

                df_meta = pd.read_csv(meta_path)
                fmin = df_meta["frame_min"].to_numpy()[split_idx].astype(np.float32)
                fmax = df_meta["frame_max"].to_numpy()[split_idx].astype(np.float32)
                scale = (fmax - fmin)[:, None]
                X_ref = X_ref * scale + fmin[:, None]
        print(
            f"[RECV10] Compare mode enabled with split={args.compare_split}, frames={X_ref.shape[0]}"
        )
    elif args.compare_realtime:
        print("[RECV10] Compare mode enabled with realtime originals from packets.")

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.port))
    sock.settimeout(float(args.socket_timeout_s))

    print(f"[RECV10] Listening udp://{args.bind_ip}:{args.port} | {tag}")

    packets = 0
    frames = 0
    bytes_total = 0

    last_seq = None
    seq_gaps = 0
    out_of_order = 0

    wf = []  # store recent reconstructed PSD frames (in chosen scale)
    wf_orig = []  # optional compare-mode originals
    last_psd = None
    last_orig = None

    t0 = time.perf_counter()
    last_recv = time.perf_counter()

    try:
        while True:
            if args.max_packets > 0 and packets >= args.max_packets:
                break
            if (
                args.idle_stop_s > 0
                and (time.perf_counter() - last_recv) > args.idle_stop_s
                and packets > 0
            ):
                print("[RECV10] Idle stop.")
                break

            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue

            last_recv = time.perf_counter()
            packets += 1
            bytes_total += len(data)

            (
                hdr,
                mu_block,
                frame_min,
                frame_max,
                x_orig_pkt,
            ) = packmod.decode_packet_to_mu_and_minmax(data)
            L = int(mu_block.shape[0])
            frames += L

            # Seq checks
            if last_seq is not None:
                if hdr.seq < last_seq:
                    out_of_order += 1
                elif hdr.seq > last_seq + 1:
                    seq_gaps += int(hdr.seq - last_seq - 1)
            last_seq = hdr.seq

            # Dequant mu
            mu_f = (mu_block.astype(np.float32) - float(out_zp)) * float(
                out_scale
            )  # (L,32)

            # Decode to normalized PSD
            x_norm = decoder(mu_f, training=False).numpy()[:, :, 0]  # (L,1024)

            # Convert to original scale if requested or required by per_frame_minmax
            if frame_min is not None and frame_max is not None and (
                args.invert_norm_to_original or norm_mode == "per_frame_minmax"
            ):
                scale = (frame_max - frame_min)[:, None]
                x_out = x_norm * scale + frame_min[:, None]
                scale_mode = "per_frame_minmax"
            elif args.invert_norm_to_original:
                x_out = invert_global_minmax(x_norm, gmin=gmin, gmax=gmax)
                scale_mode = "global_minmax"
            else:
                x_out = x_norm
                scale_mode = "normalized"

            last_psd = x_out[-1].copy()

            # Waterfall ring buffer
            wf.append(x_out)
            if args.compare_realtime and x_orig_pkt is not None:
                x_orig = x_orig_pkt
                last_orig = x_orig[-1].copy()
                wf_orig.append(x_orig)
            elif compare_enabled and X_ref is not None and ref_cursor < X_ref.shape[0]:
                n_take = min(L, X_ref.shape[0] - ref_cursor)
                x_orig = X_ref[ref_cursor : ref_cursor + n_take]
                ref_cursor += n_take
                last_orig = x_orig[-1].copy()
                wf_orig.append(x_orig)
            else:
                x_orig = None

            if x_orig is not None and args.plot_every_packets > 0 and (
                packets % args.plot_every_packets == 0
            ):
                plt.figure(figsize=(10, 3))
                plt.plot(x_orig[0], label="orig")
                plt.plot(x_out[0], label="recon")
                plt.title(f"Overlay packet {packets} (frame 0)")
                plt.xlabel("bin")
                plt.ylabel(
                    "value"
                    if not args.invert_norm_to_original
                    else "original scale"
                )
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / f"overlay_pkt{packets:06d}.png", dpi=180)
                plt.close()
            # trim by frame count
            while sum(a.shape[0] for a in wf) > args.waterfall_max_frames:
                wf.pop(0)
            while (
                compare_enabled
                and sum(a.shape[0] for a in wf_orig) > args.waterfall_max_frames
            ):
                wf_orig.pop(0)

            # Save plots periodically
            if args.save_every_packets > 0 and (packets % args.save_every_packets == 0):
                W = np.vstack(wf)

                plt.figure(figsize=(10, 5))
                plt.imshow(W, aspect="auto", origin="lower", interpolation="nearest")
                plt.title("Waterfall RECON (recent frames)")
                plt.xlabel("bin")
                plt.ylabel("frame")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(out_dir / "waterfall_recon.png", dpi=180)
                plt.close()

                if compare_enabled and wf_orig:
                    W0 = np.vstack(wf_orig)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(
                        W0, aspect="auto", origin="lower", interpolation="nearest"
                    )
                    plt.title("Waterfall ORIG (recent frames)")
                    plt.xlabel("bin")
                    plt.ylabel("frame")
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(out_dir / "waterfall_orig.png", dpi=180)
                    plt.close()

                plt.figure(figsize=(10, 3))
                plt.plot(last_psd)
                plt.title("Last reconstructed PSD (production)")
                plt.xlabel("bin")
                plt.ylabel(
                    "value" if not args.invert_norm_to_original else "original scale"
                )
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(out_dir / "psd_last.png", dpi=180)
                plt.close()

                np.save(out_dir / "recon_last_psd.npy", last_psd)
                if compare_enabled and last_orig is not None:
                    np.save(out_dir / "orig_last_psd.npy", last_orig)

                ram_mb = (
                    float(psutil.Process().memory_info().rss) / (1024 * 1024)
                    if psutil is not None
                    else None
                )
                metrics = {
                    "tag": tag,
                    "packets": int(packets),
                    "frames": int(frames),
                    "bytes_total": int(bytes_total),
                    "avg_bytes_per_frame": float(bytes_total / max(1, frames)),
                    "frames_per_second": float(
                        frames / max(1e-9, (time.perf_counter() - t0))
                    ),
                    "seq_gaps": int(seq_gaps),
                    "out_of_order": int(out_of_order),
                    "dequant": {
                        "out_scale": float(out_scale),
                        "out_zero_point": int(out_zp),
                    },
                    "scale": scale_mode,
                }
                if ram_mb is not None:
                    metrics["ram_mb"] = ram_mb
                if args.compare_realtime:
                    metrics["compare_mode"] = "realtime"
                elif compare_enabled and X_ref is not None:
                    metrics["compare_mode"] = "split"
                    metrics["compare_split"] = args.compare_split
                    metrics["compare_frames_used"] = int(
                        min(ref_cursor, X_ref.shape[0])
                    )
                (out_dir / "udp_prod_metrics.json").write_text(
                    json.dumps(metrics, indent=2)
                )
                print("[RECV10] Saved plots/metrics:", out_dir)

    except KeyboardInterrupt:
        print("[RECV10] Ctrl+C, stopping...")

    dt = time.perf_counter() - t0
    print(
        f"[RECV10] DONE packets={packets} frames={frames} bytes={bytes_total} "
        f"| frames/s={frames / max(1e-9, dt):.1f} | avg bytes/frame={bytes_total / max(1, frames):.3f}"
    )
    print("[RECV10] seq_gaps:", seq_gaps, "| out_of_order:", out_of_order)
    print("[RECV10] outputs:", out_dir)


if __name__ == "__main__":
    main()
