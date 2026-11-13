#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub>=0.24",
#     "tokenizers>=0.19",
# ]
# ///
"""Stream a Qwen3-0.6B continuation of PREFIX from the assembly engine.

usage: ./inference.py "PREFIX"

The prefix is printed, then the model's continuation is streamed after it.
Weights are downloaded from Hugging Face if missing, and the engine (qwen3.s)
is assembled, linked and given the weights if it is missing or out of date.

The engine is started once, fed BOS followed by the prefix, and left running:
it streams tokens until it emits an end-of-text token or fills its 8192-token
context, and this script does not stop it early.
"""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "Qwen3-0.6B"
BUILD_DIR = ROOT / "build"
ENGINE = BUILD_DIR / "qwen3"
SOURCE = ROOT / "qwen3.s"
ATTACH = ROOT / "tools" / "attach_weights.py"
WEIGHTS = MODEL_DIR / "model.safetensors"
REPO_ID = "Qwen/Qwen3-0.6B"
REPO_FILES = ["config.json", "tokenizer.json", "model.safetensors"]
MAX_CONTEXT = 8192
EOS_TOKENS = (151643, 151645)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def ensure_weights():
    missing = [f for f in REPO_FILES if not (MODEL_DIR / f).exists()]
    if not missing:
        return
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in missing:
        log(f"downloading {REPO_ID}/{name} ...")
        hf_hub_download(REPO_ID, name, local_dir=MODEL_DIR)


def engine_is_stale():
    if not ENGINE.exists():
        return True
    built = ENGINE.stat().st_mtime
    return any(p.stat().st_mtime > built for p in (SOURCE, ATTACH, WEIGHTS))


def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def build_engine():
    log("building engine ...")
    BUILD_DIR.mkdir(exist_ok=True)
    obj = BUILD_DIR / "qwen3.o"
    base = BUILD_DIR / "qwen3-base"
    run(["as", "--64", "-o", obj, SOURCE])
    run(["ld", "-static", "-nostdlib", "-o", base, obj])
    run([sys.executable, ATTACH, base, MODEL_DIR, ENGINE])


def emit(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def stream_tokens(proc, tok):
    """Read u32 tokens from the engine and print the text they decode to.

    Uses the usual prefix/read offset scheme so that multi-byte characters split
    across tokens are only printed once complete.
    """
    out_ids = []
    prefix_off = 0
    read_off = 0
    while True:
        raw = proc.stdout.read(4)
        if len(raw) < 4:
            break
        (token,) = struct.unpack("<I", raw)
        if token in EOS_TOKENS:
            break
        out_ids.append(token)
        before = tok.decode(out_ids[prefix_off:read_off])
        after = tok.decode(out_ids[prefix_off:])
        if len(after) > len(before) and not after.endswith("�"):
            emit(after[len(before):])
            prefix_off = read_off
            read_off = len(out_ids)
    if read_off < len(out_ids):
        before = tok.decode(out_ids[prefix_off:read_off])
        emit(tok.decode(out_ids[prefix_off:])[len(before):])


def main():
    if len(sys.argv) != 2:
        print(__doc__.split("\n\n")[1].strip(), file=sys.stderr)
        return 2
    prefix_text = sys.argv[1]

    ensure_weights()
    if engine_is_stale():
        build_engine()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    bos = json.loads((MODEL_DIR / "config.json").read_text())["bos_token_id"]
    prompt_ids = [bos] + tok.encode(prefix_text, add_special_tokens=False).ids
    if len(prompt_ids) > MAX_CONTEXT:
        log(f"prefix is {len(prompt_ids)} tokens; the context is {MAX_CONTEXT}")
        return 1

    proc = subprocess.Popen([str(ENGINE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        # The engine reads its whole prefix until stdin closes, then generates.
        proc.stdin.write(struct.pack(f"<{len(prompt_ids)}I", *prompt_ids))
        proc.stdin.close()
        emit(prefix_text)
        stream_tokens(proc, tok)
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        return 130
    except BrokenPipeError:
        proc.kill()
        proc.wait()
        return 0
    rc = proc.wait()
    emit("\n")
    if rc != 0:
        log(f"engine exited with status {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
