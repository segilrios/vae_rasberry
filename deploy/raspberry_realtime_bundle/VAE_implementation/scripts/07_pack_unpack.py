#!/usr/bin/env python3
"""
07_pack_unpack.py — Packet format + pack/unpack for mu_int8 streams (32 dims).

This file is the "wire format" building block for the edge->server UDP pipeline.

Concept:
- You have mu_int8 frames: shape (L, 32) int8, for a time-contiguous block.
- You encode the block as:
    [L (uint8)] + mu0 (32 int8 bytes) + deltas ((L-1)*32 int8 bytes)
  where delta[t] = clip(mu[t] - mu[t-1], -128..127) stored as int8.
- You compress the block bytes with zlib.
- You wrap it with a small binary header for UDP streaming.

Header (big-endian) — 24 bytes total:
- magic      4s   = b"KLP1"
- version    u8   = 1
- flags      u8   bit0=keyframe (1 if block starts absolute mu0)
- reserved   u16  = 0
- seq        u32  packet sequence number
- ts_ms      u64  timestamp in ms
- payload_len u32 length of compressed payload in bytes
Then: payload bytes (zlib compressed block)

Usage:
- Import functions from this file in your UDP sender/receiver.
- Or run self-test:
    python VAE_implementation/scripts/07_pack_unpack.py --self_test
"""

import argparse
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


MAGIC = b"KLP1"
VERSION = 1

# Big-endian header:
# 4s magic, B version, B flags, H reserved, I seq, Q ts_ms, I payload_len
HDR_STRUCT = struct.Struct(">4sBBHIQI")
HDR_SIZE = HDR_STRUCT.size  # 24 bytes

FLAG_KEYFRAME = 1 << 0  # block contains absolute mu0


@dataclass
class PacketHeader:
    version: int
    flags: int
    seq: int
    ts_ms: int
    payload_len: int


# -----------------------------
# Block coding (mu_int8 <-> bytes)
# -----------------------------
def encode_mu_block(mu_block: np.ndarray) -> bytes:
    """
    mu_block: (L, 32) int8
    returns raw bytes: [L][mu0][deltas]
    """
    if not isinstance(mu_block, np.ndarray):
        mu_block = np.asarray(mu_block)

    if mu_block.ndim != 2 or mu_block.shape[1] != 32:
        raise ValueError(f"mu_block must have shape (L,32). Got {mu_block.shape}")

    if mu_block.dtype != np.int8:
        mu_block = mu_block.astype(np.int8, copy=False)

    L = int(mu_block.shape[0])
    if L <= 0:
        raise ValueError("mu_block must have L>=1")
    if L > 255:
        raise ValueError("L must be <=255 per packet. Split the stream into smaller blocks.")

    mu0 = mu_block[0].astype(np.int8, copy=False)

    if L == 1:
        return bytes([L]) + mu0.tobytes()

    # delta in int16 then clip to int8
    d = mu_block[1:].astype(np.int16) - mu_block[:-1].astype(np.int16)  # (L-1,32) int16
    d8 = np.clip(d, -128, 127).astype(np.int8)

    return bytes([L]) + mu0.tobytes() + d8.tobytes()


def decode_mu_block(raw: bytes) -> np.ndarray:
    """
    raw bytes: [L][mu0][deltas]
    returns mu_block: (L,32) int8
    """
    if len(raw) < 1 + 32:
        raise ValueError("raw block too small")

    L = raw[0]
    if L == 0:
        raise ValueError("Invalid L=0")

    expected = 1 + 32 + (L - 1) * 32
    if len(raw) < expected:
        raise ValueError(f"Truncated block: expected {expected} bytes, got {len(raw)}")

    mu0 = np.frombuffer(raw[1:33], dtype=np.int8).astype(np.int16)  # int16 for cumulative sum
    out = np.zeros((L, 32), dtype=np.int16)
    out[0] = mu0

    if L > 1:
        dbytes = raw[33:expected]
        d = np.frombuffer(dbytes, dtype=np.int8).reshape((L - 1, 32)).astype(np.int16)
        for t in range(1, L):
            out[t] = np.clip(out[t - 1] + d[t - 1], -128, 127)

    return out.astype(np.int8)


# -----------------------------
# Packet coding (header + zlib payload)
# -----------------------------
def pack_packet(mu_block: np.ndarray,
                seq: int,
                ts_ms: Optional[int] = None,
                zlib_level: int = 1,
                keyframe: bool = True) -> bytes:
    """
    Build a packet:
      header + zlib( encode_mu_block(mu_block) )

    - seq: u32
    - ts_ms: unix epoch ms (if None, computed)
    - zlib_level: 1..9 recommended (1 is fastest and worked well in your benchmark)
    - keyframe: if True sets bit0 in flags

    Returns: bytes ready to send over UDP.
    """
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)

    raw_block = encode_mu_block(mu_block)
    payload = zlib.compress(raw_block, level=zlib_level)

    flags = FLAG_KEYFRAME if keyframe else 0
    reserved = 0

    header = HDR_STRUCT.pack(MAGIC, VERSION, flags, reserved, int(seq) & 0xFFFFFFFF, int(ts_ms) & 0xFFFFFFFFFFFFFFFF, len(payload))
    return header + payload


def unpack_packet(pkt: bytes) -> Tuple[PacketHeader, bytes]:
    """
    Parse header and extract compressed payload bytes.
    Returns: (PacketHeader, payload_compressed)
    """
    if len(pkt) < HDR_SIZE:
        raise ValueError("packet too small")

    magic, ver, flags, _reserved, seq, ts_ms, payload_len = HDR_STRUCT.unpack_from(pkt, 0)
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic}")
    if ver != VERSION:
        raise ValueError(f"Unsupported version: {ver}")

    if len(pkt) < HDR_SIZE + payload_len:
        raise ValueError("Truncated packet payload")

    payload = pkt[HDR_SIZE:HDR_SIZE + payload_len]
    hdr = PacketHeader(version=ver, flags=flags, seq=seq, ts_ms=ts_ms, payload_len=payload_len)
    return hdr, payload


def decode_packet_to_mu(pkt: bytes) -> Tuple[PacketHeader, np.ndarray]:
    """
    Full decode: packet -> zlib decompress -> mu_block int8 (L,32)
    """
    hdr, payload = unpack_packet(pkt)
    raw = zlib.decompress(payload)
    mu = decode_mu_block(raw)
    return hdr, mu


# -----------------------------
# Helpers for splitting streams into blocks
# -----------------------------
def split_into_blocks(mu_stream: np.ndarray, block_len: int) -> list:
    """
    mu_stream: (N,32) int8
    block_len: L per packet (<=255)
    returns list of mu_blocks
    """
    if block_len <= 0 or block_len > 255:
        raise ValueError("block_len must be in [1,255]")

    if mu_stream.ndim != 2 or mu_stream.shape[1] != 32:
        raise ValueError(f"mu_stream must have shape (N,32). Got {mu_stream.shape}")

    blocks = []
    for i in range(0, mu_stream.shape[0], block_len):
        blocks.append(mu_stream[i:i + block_len].astype(np.int8, copy=False))
    return blocks


# -----------------------------
# Self-test
# -----------------------------
def self_test(n_frames: int = 120, block_len: int = 30, zlib_level: int = 1, seed: int = 2026) -> None:
    rng = np.random.default_rng(seed)
    mu = rng.integers(low=-128, high=128, size=(n_frames, 32), dtype=np.int16)
    mu = mu.astype(np.int8)

    blocks = split_into_blocks(mu, block_len=block_len)

    packets = []
    for i, b in enumerate(blocks):
        pkt = pack_packet(b, seq=i, zlib_level=zlib_level, keyframe=True)
        packets.append(pkt)

    recovered = []
    for pkt in packets:
        hdr, mub = decode_packet_to_mu(pkt)
        recovered.append(mub)

    mu_rec = np.vstack(recovered)

    # Note: this should be identical because we encode absolute mu0 and deltas (with clipping).
    # However, because delta is clipped, if mu changes by >127 in one step, reconstruction may differ.
    # With random int8 jumps, that can happen. To make the test strict, generate a smooth random walk.
    print("[SELFTEST] Random mu test: exact equality:", np.array_equal(mu, mu_rec))

    # Now strict test with bounded step (random walk with small steps)
    steps = rng.integers(low=-3, high=4, size=(n_frames, 32), dtype=np.int16)
    walk = np.zeros((n_frames, 32), dtype=np.int16)
    for t in range(1, n_frames):
        walk[t] = np.clip(walk[t - 1] + steps[t], -128, 127)
    mu2 = walk.astype(np.int8)

    blocks2 = split_into_blocks(mu2, block_len=block_len)
    packets2 = [pack_packet(b, seq=i, zlib_level=zlib_level, keyframe=True) for i, b in enumerate(blocks2)]
    mu2_rec = np.vstack([decode_packet_to_mu(p)[1] for p in packets2])

    print("[SELFTEST] Random-walk mu test: exact equality:", np.array_equal(mu2, mu2_rec))

    # Size stats
    raw_bytes = n_frames * 32
    pkt_bytes = sum(len(p) for p in packets2)
    print(f"[SELFTEST] Raw mu bytes total: {raw_bytes}")
    print(f"[SELFTEST] Packed+zlib bytes total: {pkt_bytes}  -> {pkt_bytes/n_frames:.3f} bytes/frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--n_frames", type=int, default=120)
    ap.add_argument("--block_len", type=int, default=30)
    ap.add_argument("--zlib_level", type=int, default=1)
    args = ap.parse_args()

    if args.self_test:
        self_test(n_frames=args.n_frames, block_len=args.block_len, zlib_level=args.zlib_level)
    else:
        print("This module is meant to be imported by UDP sender/receiver scripts.")
        print("Run self-test with: python VAE_implementation/scripts/07_pack_unpack.py --self_test")


if __name__ == "__main__":
    main()