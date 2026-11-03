#!/usr/bin/env python3
"""Pack the M8 mix into the token arrays the training scripts read (docs/roadmap-models.md section 7).

    python3 examples/qwen/m8_pack.py DATA_DIR [--seq 512]

Reads DATA_DIR/{train,heldout_repos,heldout_files}.jsonl (m8_mix.py) and writes DATA_DIR/packed.npz with, for each split S: `S` int32 [n, seq] (a stream of samples joined by <|endoftext|>, cut every `seq` tokens, the tail
dropped) and `S_mask` uint8 [n, seq] (1 where the loss counts). A document counts everywhere; a pair is wrapped in the Qwen2.5 chat template (system, user = the prompt, assistant = the reply, <|im_end|>) and
only the reply and its <|im_end|> count. `corpus.npz` holds `train` and `heldout` (= heldout_files) reshaped as examples/qwen/finetune.py reads them (batches of 8 x 512; no mask: that session's loss is over every position).
Counts and the sha256 of both files go to DATA_DIR/PACK.json."""
import argparse
import hashlib
import json
import os

import numpy as np

TOKENIZER = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots")
SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
ASSIST = "<|im_end|>\n<|im_start|>assistant\n"
EOT = 151643


def tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(TOKENIZER, sorted(os.listdir(TOKENIZER))[0], "tokenizer.json"))


def sample_tokens(t, r):
    """(ids, loss mask) of one record."""
    if "text" in r:
        ids = t.encode(r["text"]).ids
        return ids, [1] * len(ids)
    head = t.encode(SYSTEM + r["prompt"] + ASSIST).ids
    tail = t.encode(r["reply"] + "<|im_end|>\n").ids
    return head + tail, [0] * len(head) + [1] * len(tail)


def pack(t, rows, seq):
    ids, mask = [], []
    for r in rows:
        a, m = sample_tokens(t, r)
        ids += a + [EOT]
        mask += m + [0]
    n = len(ids) // seq
    return np.array(ids[:n * seq], dtype=np.int32).reshape(n, seq), np.array(mask[:n * seq], dtype=np.uint8).reshape(n, seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()
    t = tok()
    out, rep = {}, {"seq": a.seq, "splits": {}}
    for s in ("train", "heldout_repos", "heldout_files"):
        rows = [json.loads(l) for l in open(os.path.join(a.data, s + ".jsonl"))]
        x, m = pack(t, rows, a.seq)
        out[s], out[s + "_mask"] = x, m
        rep["splits"][s] = {"sequences": int(x.shape[0]), "tokens": int(x.size), "loss_tokens": int(m.sum())}
    np.savez(os.path.join(a.data, "packed.npz"), **out)
    nb = (out["train"].shape[0] // 8) * 8
    nh = (out["heldout_files"].shape[0] // 8) * 8
    np.savez(os.path.join(a.data, "corpus.npz"), train=out["train"][:nb].astype(np.int64), heldout=out["heldout_files"][:nh].astype(np.int64))
    rep["corpus_npz"] = {"train_sequences": nb, "heldout_sequences": nh}
    for f in ("packed.npz", "corpus.npz"):
        rep[f] = hashlib.sha256(open(os.path.join(a.data, f), "rb").read()).hexdigest()
    json.dump(rep, open(os.path.join(a.data, "PACK.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
