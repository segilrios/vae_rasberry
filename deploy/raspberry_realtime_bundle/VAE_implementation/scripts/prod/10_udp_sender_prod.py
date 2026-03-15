#!/usr/bin/env python3
"""
10_udp_sender_prod.py ??? Production UDP sender (edge/Raspberry).

Sends ONLY what is needed in production:
- packet header (KLP1) + zlib(payload)
- payload = encode_mu_block(mu_block)   (L + mu0 + deltas)

No indices, no debug overhead.

Sources:
- dataset: uses processed dataset to simulate streaming
- npy_dir: reads 1024-float arrays from a folder (one PSD per file)

Normalization:
- If you use npy_dir and PSDs are NOT normalized, set --apply_norm_from_npz
  to normalize using gmin/gmax stored in dataset_psd_1024_norm.npz

Requires on Raspberry:
- preferred: pip install tflite-runtime numpy pyyaml
- fallback: tensorflow (heavier)

Run example (sim):
  python VAE_implementation/scripts/prod/10_udp_sender_prod.py --config ... --use_global_best --dest_ip <server_ip> --port 5005 --source dataset --split test --block_len 30 --n_blocks 50
"""

import argparse
import importlib.util
import socket
import time
from pathlib import Path

import numpy as np
import yaml
from importlib.machinery import ModuleSpec

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
        try:
            import tensorflow as tf  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install `tflite-runtime` or `tensorflow` to run the UDP sender."
            ) from exc

        return tf.lite.Interpreter


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


def load_metadata(processed_dir: Path):
    meta_path = processed_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {processed_dir}")
    import pandas as pd

    return pd.read_csv(meta_path)


def load_tflite(tflite_path: Path):
    interp = _resolve_tflite_interpreter()(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    return interp, in_det, out_det


def infer_mu_int8_one(interp, in_det, out_det, x_norm_1024: np.ndarray) -> np.ndarray:
    in_scale, in_zp = in_det["quantization"]
    x = x_norm_1024[None, :, None].astype(np.float32)  # (1,1024,1)

    if in_det["dtype"] == np.int8:
        xq = np.round(x / in_scale + in_zp).astype(np.int8)
    elif in_det["dtype"] == np.uint8:
        xq = np.round(x / in_scale + in_zp).astype(np.uint8)
    else:
        xq = x.astype(in_det["dtype"])

    interp.set_tensor(in_det["index"], xq)
    interp.invoke()
    yq = interp.get_tensor(out_det["index"])[0]  # (32,)
    return yq.astype(np.int8, copy=False)


def apply_global_minmax(x: np.ndarray, gmin: float, gmax: float) -> np.ndarray:
    return (x - gmin) / (gmax - gmin + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_global_best", action="store_true")
    ap.add_argument("--run_name", default=None)

    ap.add_argument("--dest_ip", required=True)
    ap.add_argument("--port", type=int, default=5005)

    ap.add_argument("--source", choices=["dataset", "npy_dir"], default="dataset")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")

    ap.add_argument(
        "--npy_dir",
        default=None,
        help="Folder with PSD .npy files (each is (1024,) float32)",
    )
    ap.add_argument(
        "--already_normalized",
        action="store_true",
        help="If set, sender assumes PSD vectors are already normalized [0,1].",
    )
    ap.add_argument(
        "--apply_norm_from_npz",
        action="store_true",
        help="If set, normalize using gmin/gmax stored in processed npz (global_minmax).",
    )
    ap.add_argument(
        "--send_minmax",
        action="store_true",
        help="Send per-frame min/max in packets (required for per_frame_minmax reconstruction).",
    )
    ap.add_argument(
        "--source_file",
        default=None,
        help="Filter dataset source to a single source_file value from metadata.csv.",
    )

    ap.add_argument("--block_len", type=int, default=30)
    ap.add_argument(
        "--n_blocks", type=int, default=0, help="0 = send as many blocks as possible"
    )
    ap.add_argument("--zlib_level", type=int, default=1)
    ap.add_argument("--sleep_ms", type=float, default=0.0)
    ap.add_argument("--packet_interval_ms", type=float, default=0.0)
    ap.add_argument("--log_every_packets", type=int, default=10)
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
        raise FileNotFoundError(
            f"Missing TFLite encoder: {tflite_path} (run 04_export_tflite.py)."
        )

    packmod = load_pack_module()
    interp, in_det, out_det = load_tflite(tflite_path)

    normalize_mode_cfg = str(cfg.get("preprocess", {}).get("normalize_mode", "global_minmax"))

    # Normalization parameters (optional)
    gmin = gmax = None
    npz = None
    if args.apply_norm_from_npz:
        npz = load_npz(processed_dir)
        if "normalize_mode" in npz and str(npz["normalize_mode"]) != "global_minmax":
            print(
                "[SEND10] WARN: normalize_mode is not global_minmax in npz; apply_norm_from_npz may not match training."
            )
        gmin = float(npz.get("gmin", np.nan))
        gmax = float(npz.get("gmax", np.nan))
        if not np.isfinite(gmin + gmax):
            raise ValueError(
                "apply_norm_from_npz requested but gmin/gmax not found or invalid in npz."
            )
    normalize_mode = (
        str(npz.get("normalize_mode", normalize_mode_cfg)) if npz is not None else normalize_mode_cfg
    )

    # Build frame index list depending on source
    L = int(args.block_len)
    if L <= 0 or L > 255:
        raise ValueError("block_len must be in [1,255]")

    send_minmax = bool(args.send_minmax or (normalize_mode == "per_frame_minmax"))

    if args.source == "dataset":
        npz = npz or load_npz(processed_dir)
        X = npz["X"].astype(np.float32)  # (N,1024)
        tr, va, te = load_splits(processed_dir)
        idx = {"train": tr, "val": va, "test": te}[args.split].astype(np.int64)

        meta = None
        frame_min = frame_max = None
        source_file_arr = None
        if send_minmax or args.source_file:
            meta = load_metadata(processed_dir)
            if "frame_min" in meta.columns and "frame_max" in meta.columns:
                frame_min = meta["frame_min"].to_numpy().astype(np.float32)
                frame_max = meta["frame_max"].to_numpy().astype(np.float32)
            if "source_file" in meta.columns:
                source_file_arr = meta["source_file"].astype(str).to_numpy()

        if args.source_file:
            if source_file_arr is None:
                raise ValueError("metadata.csv missing source_file column.")
            keep = np.where(source_file_arr[idx] == str(args.source_file))[0]
            if keep.size == 0:
                raise ValueError(f"No frames match source_file={args.source_file}")
            idx = idx[keep]

        total_blocks = idx.shape[0] // L
        if total_blocks <= 0:
            raise ValueError(
                f"Split '{args.split}' does not have enough frames for block_len={L}."
            )

        n_blocks = (
            total_blocks
            if args.n_blocks == 0
            else min(int(args.n_blocks), total_blocks)
        )
        idx = idx[: n_blocks * L]

        def get_psd(i_global: int) -> np.ndarray:
            x = X[int(i_global)]
            # already normalized in processed dataset
            return x

        def get_minmax(i_global: int) -> tuple[float, float]:
            if frame_min is None or frame_max is None:
                raise ValueError("frame_min/frame_max missing; cannot send min/max.")
            return float(frame_min[int(i_global)]), float(frame_max[int(i_global)])

        print(
            f"[SEND10] source=dataset split={args.split} blocks={n_blocks} L={L} "
            f"{'minmax' if send_minmax else 'no-minmax'}"
        )

    else:
        if not args.npy_dir:
            raise ValueError("--npy_dir is required when --source npy_dir")
        pdir = Path(args.npy_dir)
        files = sorted(pdir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files in {pdir}")

        total_frames = len(files)
        total_blocks = total_frames // L
        n_blocks = (
            total_blocks
            if args.n_blocks == 0
            else min(int(args.n_blocks), total_blocks)
        )
        files = files[: n_blocks * L]

        def get_psd(i_global: int) -> np.ndarray:
            x = np.load(files[i_global]).astype(np.float32)
            if x.shape[0] != 1024:
                raise ValueError(
                    f"PSD file {files[i_global].name} has shape {x.shape}, expected (1024,)"
                )
            if args.already_normalized:
                return x
            if normalize_mode == "per_frame_minmax":
                xmin = float(np.min(x))
                xmax = float(np.max(x))
                return (x - xmin) / (xmax - xmin + 1e-8)
            if args.apply_norm_from_npz and gmin is not None:
                return apply_global_minmax(x, gmin, gmax)
            raise ValueError(
                "PSD not normalized. Use --already_normalized, per_frame_minmax, or --apply_norm_from_npz."
            )

        idx = np.arange(len(files), dtype=np.int64)

        def get_minmax(i_global: int) -> tuple[float, float]:
            x = np.load(files[i_global]).astype(np.float32)
            return float(np.min(x)), float(np.max(x))

        print(
            f"[SEND10] source=npy_dir frames={len(files)} blocks={n_blocks} L={L} "
            f"{'minmax' if send_minmax else 'no-minmax'}"
        )

    # UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.dest_ip, args.port)

    print(
        f"[SEND10] -> udp://{args.dest_ip}:{args.port} | {tag} | zlib={args.zlib_level}"
    )

    seq = 0
    sent_frames = 0
    sent_bytes = 0
    t0 = time.perf_counter()

    last_send_t = time.perf_counter()
    for b in range(n_blocks):
        s = b * L

        mu_block = np.zeros((L, 32), dtype=np.int8)
        mm_min = np.zeros((L,), dtype=np.float32) if send_minmax else None
        mm_max = np.zeros((L,), dtype=np.float32) if send_minmax else None

        for i in range(L):
            x_norm = get_psd(int(idx[s + i]))
            mu_block[i] = infer_mu_int8_one(interp, in_det, out_det, x_norm)
            if send_minmax and mm_min is not None and mm_max is not None:
                fmin, fmax = get_minmax(int(idx[s + i]))
                mm_min[i] = fmin
                mm_max[i] = fmax

        pkt = packmod.pack_packet(
            mu_block,
            seq=seq,
            zlib_level=args.zlib_level,
            keyframe=True,
            frame_min=mm_min,
            frame_max=mm_max,
        )
        sock.sendto(pkt, dest)

        seq += 1
        sent_frames += L
        sent_bytes += len(pkt)

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)
        if args.packet_interval_ms > 0:
            target = args.packet_interval_ms / 1000.0
            elapsed = time.perf_counter() - last_send_t
            if elapsed < target:
                time.sleep(target - elapsed)
            last_send_t = time.perf_counter()

        if args.log_every_packets > 0 and (seq % args.log_every_packets == 0):
            dt = max(1e-9, time.perf_counter() - t0)
            ram_mb = (
                float(psutil.Process().memory_info().rss) / (1024 * 1024)
                if psutil is not None
                else None
            )
            ram_txt = f" ramMB={ram_mb:.1f}" if ram_mb is not None else ""
            print(
                f"[SEND10] pkts={seq} frames={sent_frames} avgB/f={sent_bytes / max(1, sent_frames):.2f} "
                f"frames/s={sent_frames / dt:.1f} KB/s={sent_bytes / dt / 1024:.2f}{ram_txt}"
            )

    dt = time.perf_counter() - t0
    print(
        f"[SEND10] DONE blocks={seq} frames={sent_frames} bytes={sent_bytes} "
        f"| frames/s={sent_frames / dt:.1f} | KB/s={sent_bytes / dt / 1024:.2f} | avg bytes/frame={sent_bytes / max(1, sent_frames):.3f}"
    )


if __name__ == "__main__":
    main()
