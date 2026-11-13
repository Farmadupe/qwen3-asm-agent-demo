#!/usr/bin/env python3
"""Attach Qwen3-0.6B weights to the qwen3 engine executable.

usage: attach_weights.py ENGINE_ELF MODEL_DIR OUT_BINARY

ENGINE_ELF  the linked engine without weights (build/qwen3-base).
MODEL_DIR   directory with config.json and model.safetensors (BF16).
OUT_BINARY  the engine with the weight blob attached.

Output layout:
    [ENGINE_ELF bytes]
    [zero padding to a 4096-byte boundary]
    [blob]      4096-byte header, then tensors, each 64-byte aligned
    [trailer]   u64 blob_file_offset, u64 blob_size, b"QWN3TRL1"  (24 bytes)

Blob header (little endian):
    0    b"QWN3BLOB"
    8    u32 version (1), u32 nslots
    16   u32 config[8] = vocab, hidden, layers, heads, kv_heads, head_dim, inter, max_ctx
    48   u64 offset[nslots]   (blob-relative)

Slot order (must match qwen3.s):
    0 embed_tokens (V x H, bf16)      1 final norm (H, bf16)
    2 + 11*l + k for layer l:
        k=0 input_layernorm  1 q_proj  2 k_proj  3 v_proj  4 o_proj
        k=5 q_norm  6 k_norm  7 post_attention_layernorm
        k=8 gate_proj  9 up_proj  10 down_proj
    310 rope cos (MAX_CTX x 64, f32)   311 rope sin (MAX_CTX x 64, f32)

Only the standard library is used.
"""
import array
import json
import math
import os
import struct
import sys

# Qwen3-0.6B dimensions; must match qwen3.s.
VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTER = 151936, 1024, 28, 16, 8, 128, 3072
MAX_CTX = 8192
THETA = 1_000_000.0
Q_DIM = HEADS * HEAD_DIM
KV_DIM = KV_HEADS * HEAD_DIM
NSLOTS = 2 + LAYERS * 11 + 2
SLOT_COS = 2 + LAYERS * 11
SLOT_SIN = SLOT_COS + 1
HEADER_SIZE = 4096
PAGE = 4096
ALIGN = 64
BLOB_MAGIC = b"QWN3BLOB"
TRAILER_MAGIC = b"QWN3TRL1"
CHUNK = 16 << 20


def layer_specs():
    specs = [("model.embed_tokens.weight", (VOCAB, HIDDEN)), ("model.norm.weight", (HIDDEN,))]
    for l in range(LAYERS):
        p = f"model.layers.{l}."
        specs += [
            (p + "input_layernorm.weight", (HIDDEN,)),
            (p + "self_attn.q_proj.weight", (Q_DIM, HIDDEN)),
            (p + "self_attn.k_proj.weight", (KV_DIM, HIDDEN)),
            (p + "self_attn.v_proj.weight", (KV_DIM, HIDDEN)),
            (p + "self_attn.o_proj.weight", (HIDDEN, Q_DIM)),
            (p + "self_attn.q_norm.weight", (HEAD_DIM,)),
            (p + "self_attn.k_norm.weight", (HEAD_DIM,)),
            (p + "post_attention_layernorm.weight", (HIDDEN,)),
            (p + "mlp.gate_proj.weight", (INTER, HIDDEN)),
            (p + "mlp.up_proj.weight", (INTER, HIDDEN)),
            (p + "mlp.down_proj.weight", (HIDDEN, INTER)),
        ]
    assert len(specs) == SLOT_COS
    return specs


def rope_tables():
    """cos/sin tables, [MAX_CTX][HEAD_DIM/2] f32, computed in float64."""
    inv_freq = [THETA ** (-2.0 * i / HEAD_DIM) for i in range(HEAD_DIM // 2)]
    cos_t = array.array("f")
    sin_t = array.array("f")
    for p in range(MAX_CTX):
        for f in inv_freq:
            a = p * f
            cos_t.append(math.cos(a))
            sin_t.append(math.sin(a))
    if sys.byteorder != "little":
        cos_t.byteswap()
        sin_t.byteswap()
    return cos_t.tobytes(), sin_t.tobytes()


def check_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    expect = {
        "vocab_size": VOCAB, "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM, "intermediate_size": INTER, "rope_theta": THETA,
    }
    for k, v in expect.items():
        if cfg.get(k) != v:
            sys.exit(f"error: config.json {k}={cfg.get(k)!r}, engine expects {v!r}")
    if not cfg.get("tie_word_embeddings", False):
        sys.exit("error: engine assumes tied embeddings (tie_word_embeddings=true)")


def read_safetensors_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return header, 8 + hlen


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    engine, model_dir, out_path = sys.argv[1:]
    check_config(model_dir)

    with open(engine, "rb") as f:
        engine_bytes_len = os.fstat(f.fileno()).st_size
        f.seek(max(0, engine_bytes_len - 8))
        if engine_bytes_len >= 24 and f.read(8) == TRAILER_MAGIC:
            sys.exit("error: engine already has a weight blob attached; use build/qwen3-base")

    st_path = os.path.join(model_dir, "model.safetensors")
    header, data_base = read_safetensors_header(st_path)

    # Resolve every slot to (source, file offset, size) in the blob.
    sources = []  # (kind, payload) where payload is (abs file offset, nbytes) or bytes
    for name, shape in layer_specs():
        if name not in header:
            sys.exit(f"error: tensor {name} missing from safetensors")
        info = header[name]
        if info["dtype"] != "BF16":
            sys.exit(f"error: tensor {name} has dtype {info['dtype']}, expected BF16")
        if tuple(info["shape"]) != shape:
            sys.exit(f"error: tensor {name} has shape {info['shape']}, expected {list(shape)}")
        start, end = info["data_offsets"]
        n = 2
        for d in shape:
            n *= d
        if end - start != n:
            sys.exit(f"error: tensor {name} has {end - start} bytes, expected {n}")
        sources.append(("file", (data_base + start, n)))

    # Tied embeddings: the engine uses embed_tokens as the LM head.
    if "lm_head.weight" in header:
        lm = header["lm_head.weight"]
        emb = header["model.embed_tokens.weight"]
        if lm["data_offsets"][1] - lm["data_offsets"][0] != emb["data_offsets"][1] - emb["data_offsets"][0]:
            sys.exit("error: lm_head.weight does not match embed_tokens.weight")
        with open(st_path, "rb") as f:
            a0 = data_base + lm["data_offsets"][0]
            b0 = data_base + emb["data_offsets"][0]
            left = emb["data_offsets"][1] - emb["data_offsets"][0]
            off = 0
            while off < left:
                k = min(CHUNK, left - off)
                f.seek(a0 + off)
                x = f.read(k)
                f.seek(b0 + off)
                if x != f.read(k):
                    sys.exit("error: lm_head.weight differs from embed_tokens.weight (untied model)")
                off += k

    cos_bytes, sin_bytes = rope_tables()
    rope_len = MAX_CTX * (HEAD_DIM // 2) * 4
    assert len(cos_bytes) == rope_len and len(sin_bytes) == rope_len
    sources.append(("bytes", cos_bytes))
    sources.append(("bytes", sin_bytes))
    assert len(sources) == NSLOTS

    # Layout.
    offsets = []
    pos = HEADER_SIZE
    sizes = []
    for kind, payload in sources:
        pos = (pos + ALIGN - 1) & ~(ALIGN - 1)
        offsets.append(pos)
        size = payload[1] if kind == "file" else len(payload)
        sizes.append(size)
        pos += size
    blob_size = pos

    header_bytes = bytearray(HEADER_SIZE)
    header_bytes[0:8] = BLOB_MAGIC
    struct.pack_into("<II", header_bytes, 8, 1, NSLOTS)
    struct.pack_into("<8I", header_bytes, 16, VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS,
                     HEAD_DIM, INTER, MAX_CTX)
    struct.pack_into(f"<{NSLOTS}Q", header_bytes, 48, *offsets)

    engine_size = engine_bytes_len
    blob_file_off = (engine_size + PAGE - 1) & ~(PAGE - 1)

    with open(engine, "rb") as fin, open(out_path, "wb") as fout, \
            open(st_path, "rb") as fst:
        # Engine image, then padding up to the blob's page-aligned offset.
        fin.seek(0)
        while True:
            chunk = fin.read(CHUNK)
            if not chunk:
                break
            fout.write(chunk)
        fout.write(b"\0" * (blob_file_off - engine_size))

        fout.write(header_bytes)
        cur = HEADER_SIZE
        for (kind, payload), off, size in zip(sources, offsets, sizes):
            fout.write(b"\0" * (off - cur))
            if kind == "file":
                src_off, nbytes = payload
                fst.seek(src_off)
                left = nbytes
                while left:
                    chunk = fst.read(min(CHUNK, left))
                    if not chunk:
                        sys.exit("error: short read on safetensors")
                    fout.write(chunk)
                    left -= len(chunk)
            else:
                fout.write(payload)
            cur = off + size
        assert cur == blob_size

        fout.write(struct.pack("<QQ", blob_file_off, blob_size) + TRAILER_MAGIC)

    os.chmod(out_path, 0o755)
    total = blob_file_off + blob_size + 24
    print(f"wrote {out_path}: engine {engine_size} B, blob {blob_size} B at offset {blob_file_off}, "
          f"total {total} B")


if __name__ == "__main__":
    main()
