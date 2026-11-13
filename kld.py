#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub>=0.24",
#     "tokenizers>=0.19",
#     "transformers>=4.56",
#     "torch>=2.2",
#     "pyarrow>=14",
#     "numpy",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Mean KL divergence (nats) of the assembly engine's next-token distribution
from the Hugging Face transformers Qwen3-0.6B reference.

usage: ./kld.py

Evaluates 16 non-overlapping 512-token windows from the start of the
WikiText-2 (raw) test split, joined with "\\n\\n" and tokenized with no special
tokens. For every one of the 16*512 positions it computes

    KL(P_ref || P_engine) = sum_v p_ref(v) * (log p_ref(v) - log p_engine(v))

and prints the mean over all positions, in nats, as the only output.

The reference is transformers' Qwen3ForCausalLM run in float32 on the same BF16
weights. The engine is run with its logits flag (argv[2] = 1), which makes it
write the f32 logits of every position to stderr; both sets of logits are
turned into log-probabilities here in float64.

Weights and WikiText are downloaded and the engine is built if missing.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "Qwen3-0.6B"
DATA_DIR = ROOT / "data" / "wikitext"
BUILD_DIR = ROOT / "build"
ENGINE = BUILD_DIR / "qwen3"
SOURCE = ROOT / "qwen3.s"
ATTACH = ROOT / "tools" / "attach_weights.py"
WEIGHTS = MODEL_DIR / "model.safetensors"
REPO_ID = "Qwen/Qwen3-0.6B"
REPO_FILES = ["config.json", "tokenizer.json", "model.safetensors"]
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"

N_SEQ = 16
SEQ_LEN = 512
VOCAB = 151936
KL_CHUNK = 64  # positions per float64 KL block (bounds memory)


def ensure_weights():
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in REPO_FILES:
        if not (MODEL_DIR / name).exists():
            hf_hub_download(REPO_ID, name, local_dir=MODEL_DIR)


def ensure_wikitext():
    path = DATA_DIR / WIKITEXT_FILE
    if not path.exists():
        from huggingface_hub import hf_hub_download

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        hf_hub_download(WIKITEXT_REPO, WIKITEXT_FILE, repo_type="dataset", local_dir=DATA_DIR)
    return path


def ensure_engine():
    if ENGINE.exists():
        built = ENGINE.stat().st_mtime
        if all(p.stat().st_mtime <= built for p in (SOURCE, ATTACH, WEIGHTS)):
            return
    BUILD_DIR.mkdir(exist_ok=True)
    obj = BUILD_DIR / "qwen3.o"
    base = BUILD_DIR / "qwen3-base"
    run = lambda cmd, **kw: subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)
    run(["as", "--64", "-o", obj, SOURCE])
    run(["ld", "-static", "-nostdlib", "-o", base, obj])
    run([sys.executable, ATTACH, base, MODEL_DIR, ENGINE], stdout=subprocess.DEVNULL)


def load_sequences(parquet_path):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    texts = pq.read_table(parquet_path, columns=["text"]).column("text").to_pylist()
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    ids = tok.encode("\n\n".join(texts), add_special_tokens=False).ids
    need = N_SEQ * SEQ_LEN
    if len(ids) < need:
        sys.exit(f"error: wikitext has {len(ids)} tokens, need {need}")
    return np.asarray(ids[:need], dtype=np.uint32).reshape(N_SEQ, SEQ_LEN)


def engine_logits(seq):
    """f32 logits [SEQ_LEN, VOCAB] for every position of seq, from the engine."""
    proc = subprocess.Popen(
        [str(ENGINE), "0", "1"],  # generate nothing; emit logits on stderr
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    proc.stdin.write(seq.astype("<u4").tobytes())
    proc.stdin.close()
    out = np.empty(SEQ_LEN * VOCAB, dtype="<f4")
    buf = memoryview(out).cast("B")
    got = 0
    while got < len(buf):
        n = proc.stderr.readinto(buf[got:])
        if not n:
            break
        got += n
    extra = proc.stderr.read()
    rc = proc.wait()
    if rc != 0 or got != len(buf) or extra:
        # An engine error message, if any, is the only thing it wrote to stderr.
        msg = (bytes(buf[:got]) + extra)[:4096].decode(errors="replace").strip()
        sys.exit(f"error: engine exited with status {rc} after {got + len(extra)} of "
                 f"{len(buf)} logit bytes: {msg}")
    return torch.from_numpy(out.reshape(SEQ_LEN, VOCAB))


def main():
    ensure_weights()
    parquet_path = ensure_wikitext()
    ensure_engine()
    seqs = load_sequences(parquet_path)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    total = 0.0
    count = 0
    with torch.inference_mode():
        for seq in seqs:
            q_logits = engine_logits(seq)
            input_ids = torch.from_numpy(seq.astype(np.int64))[None]
            p_logits = model(input_ids).logits[0].float()
            assert p_logits.shape == q_logits.shape
            for i in range(0, SEQ_LEN, KL_CHUNK):
                lp = torch.log_softmax(p_logits[i:i + KL_CHUNK].double(), dim=-1)
                lq = torch.log_softmax(q_logits[i:i + KL_CHUNK].double(), dim=-1)
                kl = (lp.exp() * (lp - lq)).sum(dim=-1)
                total += kl.sum().item()
                count += kl.numel()
            del p_logits, q_logits

    print(total / count)


if __name__ == "__main__":
    main()
