#!/usr/bin/env python3
"""The M8 training run (docs/roadmap-models.md section 7.4): one peft LoRA configuration on the Phase 1 mix. Phase 2 of the experiment; the configuration is fixed here before any tuned result exists.

    python3 examples/qwen/m8_train.py DATA_DIR OUT_DIR --primer FILE [--primer-samples 240] [--lr 2e-4] [--eval-every 20] [--steps-limit N]

Qwen2.5-1.5B-Instruct (float32 on the device), LoRA r=16 alpha=32 on q, k, v, o, gate, up, down, AdamW (0.9, 0.999, no weight decay), linear warm-up over 5% of the steps then cosine to 0, ONE epoch.
The stream of micro-steps is, in a fixed shuffled order (seed 20261103):
  short   4 sequences of 512 tokens from packed.npz's train split (documents: every token counts; pairs wrapped in the chat template: only the reply counts)
  long    one pair (instruction or fix) with the primer in front of its prompt, exactly as the agent sends it (user turn = primer + prompt), about 11.5 k tokens, the loss on the reply only;
          `--primer-samples` of them, spread evenly through the stream (this is what makes the tuned model see the primer-in-context prompts the benchmark gives it); gradient checkpointing for these
An optimizer step accumulates micro-steps until 16,384 tokens have been seen (a long sample weighs its tokens / 16,384), so there are about 170 steps for the short stream plus the long samples' share.
Every `--eval-every` optimizer steps (and at step 0 and the end) it measures the loss on: heldout_repos (64 sequences), heldout_files (64) and 6 held-out pairs with the primer; adapter checkpoints are saved to OUT_DIR/step<N>/.
The checkpoint to use is chosen afterwards by the held-out-repos loss alone (`--choose`), never by a benchmark score. Writes OUT_DIR/train.json (every micro-step loss, the evaluations, seconds, the memory).

Memory (measured, 64 GB machine): the live tensors are about 6.5 GB at every micro-step (`current_gb`), but the MPS caching allocator keeps every block it has ever taken (the short and the long micro-steps ask for different sizes, an evaluation for others), so the driver memory
(`driver_gb`) only grew: 32 GB after the first evaluation, 42 GB after six micro-steps, 75 GB in the first run, which then swapped and went from 1.6 s to 55-91 s per micro-step. `torch.mps.empty_cache()` after every micro-step and every evaluation holds it at
about 15 GB (short) to 25 GB (long) at the peak; it does not change a number (the losses are bit-identical). A guard aborts cleanly (adapter saved to OUT_DIR/abort_step<N>) when the process's driver memory exceeds `--memory-limit-gb` (40) or the swap in use grows by more than `--swap-growth-gb` (2).

NOT SAVED: the optimizer state (Adam's moments) and the position in the micro-step stream. There is no resume: loading a checkpoint and continuing would restart Adam from zero moments, which is not the pre-registered run."""
import argparse
import ctypes
import ctypes.util
import glob
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AttentionInterface, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_pack as K  # noqa: E402

SEED = 20261103


CHUNK_Q = 1024


def _chunked_attention(module, q, k, v, mask, dropout=0.0, scaling=None, **kw):
    """Causal attention for a long sequence in blocks of CHUNK_Q query rows (batch of one, no padding: the long samples). The library's sdpa materialises the whole [heads, T, T] score matrix in float32,
    which for T = 10,800 took 62 GB of driver memory in a trial (4 k: 9 GB, 8 k: 35 GB); a block holds [heads, 1024, up to T]."""
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
    T = q.shape[2]
    if T <= 2048:
        return sdpa_attention_forward(module, q, k, v, mask, dropout=dropout, scaling=scaling, **kw)
    k, v = repeat_kv(k, module.num_key_value_groups), repeat_kv(v, module.num_key_value_groups)
    outs = []
    for i in range(0, T, CHUNK_Q):
        j = min(T, i + CHUNK_Q)
        m = torch.ones(j - i, j, dtype=torch.bool, device=q.device).tril(diagonal=i)
        outs.append(torch.nn.functional.scaled_dot_product_attention(q[:, :, i:j], k[:, :, :j], v[:, :, :j], attn_mask=m, scale=scaling))
    return torch.cat(outs, 2).transpose(1, 2).contiguous(), None


AttentionInterface.register("sdpa", _chunked_attention)


def masked_loss(model, x, m):
    """Mean cross-entropy over the positions the mask marks. The output layer is applied to those positions only: a full [11,000 x 151,936] float32 logits matrix (6.6 GB, with its gradient and the
    float copy about 20 GB) is what made the first trial of an 11 k-token sample take 71 to 77 GB of driver memory on a 64 GB machine."""
    inner = model.base_model.model
    h = inner.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1]
    tgt = x[:, 1:]
    w = m[:, 1:].bool()
    logits = inner.lm_head(h[w]).float()
    loss = torch.nn.functional.cross_entropy(logits, tgt[w], reduction="mean")
    return loss, int(w.sum().item())


def swap_used_gb():
    """Swap in use (macOS vm.swapusage: xsw_usage {total, avail, used: u64; pagesize: u32; encrypted: bool}), read through sysctlbyname; 0.0 where it is not available."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        buf = ctypes.create_string_buffer(32)
        n = ctypes.c_size_t(32)
        if libc.sysctlbyname(b"vm.swapusage", buf, ctypes.byref(n), None, ctypes.c_size_t(0)) != 0:
            return 0.0
        return int.from_bytes(buf.raw[16:24], "little") / 1e9
    except Exception:
        return 0.0


def long_sample(tok, primer, rec):
    """(ids, mask) of a pair with the primer in front of its prompt, as the agent sends it: the user turn is primer + prompt."""
    head = tok.encode(K.SYSTEM + primer + rec["prompt"] + K.ASSIST).ids
    tail = tok.encode(rec["reply"] + "<|im_end|>\n").ids
    return head + tail, [0] * len(head) + [1] * len(tail)


def evaluate(model, dev, z, long_eval):
    model.eval()
    out = {}
    with torch.no_grad():
        for s in ("heldout_repos", "heldout_files"):
            x, m = z[s][:64], z[s + "_mask"][:64]
            ls = []
            for i in range(0, len(x), 4):
                ls.append(masked_loss(model, torch.tensor(x[i:i + 4], dtype=torch.long, device=dev), torch.tensor(m[i:i + 4], device=dev))[0].item())
            out[s] = float(np.mean(ls))
        ls = []
        for ids, mask in long_eval:
            ls.append(masked_loss(model, torch.tensor([ids], device=dev), torch.tensor([mask], device=dev))[0].item())
        out["heldout_pairs_with_primer"] = float(np.mean(ls))
    model.train()
    if str(dev).startswith("mps"):
        torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--primer", required=True)
    ap.add_argument("--primer-samples", type=int, default=240)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--tokens-per-step", type=int, default=16384)
    ap.add_argument("--steps-limit", type=int, default=0, help="stop after this many optimizer steps (a trial); 0: the whole epoch")
    ap.add_argument("--memory-limit-gb", type=float, default=40.0, help="abort cleanly (after saving an adapter checkpoint) when the process's own MPS driver memory exceeds this; 0: no guard")
    ap.add_argument("--swap-growth-gb", type=float, default=2.0, help="abort cleanly when the swap in use grows by more than this over its value at the start; 0: no guard")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model", default="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
    ap.add_argument("--choose", action="store_true", help="print the checkpoint with the lowest held-out-repos loss of OUT/train.json and exit")
    a = ap.parse_args()
    if a.choose:
        r = json.load(open(os.path.join(a.out, "train.json")))
        best = min(r["evals"], key=lambda e: e["heldout_repos"])
        print(json.dumps(best))
        return
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    z = np.load(os.path.join(a.data, "packed.npz"))
    tok = K.tok()
    primer = open(a.primer).read()
    rng = random.Random(SEED)
    pairs = [json.loads(l) for l in open(os.path.join(a.data, "train.jsonl"))]
    pairs = [r for r in pairs if "prompt" in r]
    rng.shuffle(pairs)
    long_train = [long_sample(tok, primer, r) for r in pairs[:a.primer_samples]]
    hp = [json.loads(l) for l in open(os.path.join(a.data, "heldout_repos.jsonl"))]
    hp = sorted((r for r in hp if "prompt" in r), key=lambda r: r["id"])
    step = max(1, len(hp) // 6)
    long_eval = [long_sample(tok, primer, r) for r in hp[::step][:6]]
    # the micro-step stream
    X, M = z["train"], z["train_mask"]
    order = list(range(0, len(X) - 3, 4))
    rng.shuffle(order)
    stream = [("short", i) for i in order]
    for k in range(len(long_train)):
        stream.insert(int((k + 0.5) * len(stream) / len(long_train)) + k, ("long", k))
    d = glob.glob(os.path.join(os.path.expanduser(a.model), "snapshots", "*"))
    model = AutoModelForCausalLM.from_pretrained(d[0] if d else os.path.expanduser(a.model), dtype=torch.float32)
    torch.manual_seed(7)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")).to(dev)
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    total_tokens = sum(4 * 512 if k == "short" else len(long_train[i][0]) for k, i in stream)
    steps = max(1, round(total_tokens / a.tokens_per_step))
    warm = max(1, round(0.05 * steps))
    lr_at = lambda t: a.lr * (t / warm if t <= warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm))))
    res = {"config": {"lr": a.lr, "r": 16, "alpha": 32, "tokens_per_step": a.tokens_per_step, "optimizer_steps": steps, "micro_steps": len(stream), "long_samples": len(long_train),
                      "long_sample_tokens_mean": float(np.mean([len(x[0]) for x in long_train])), "long_sample_tokens_max": max(len(x[0]) for x in long_train), "epochs": 1,
                      "short_sequences": len(order) * 4, "trainable": sum(p.numel() for p in params), "seed": SEED, "primer_sha256": hashlib.sha256(primer.encode()).hexdigest()},
           "evals": [], "micro": []}
    model.train()
    e = evaluate(model, dev, z, long_eval)
    e["step"] = 0
    res["evals"].append(e)
    model.save_pretrained(os.path.join(a.out, "step0"))
    print("step 0", e, f"({steps} optimizer steps planned)", flush=True)
    opt.zero_grad()
    seen, t_step, acc = 0, 0, 0
    t0 = time.time()
    swap0 = swap_used_gb()
    for kind, idx in stream:
        if kind == "short":
            x, m = torch.tensor(X[idx:idx + 4], dtype=torch.long, device=dev), torch.tensor(M[idx:idx + 4], device=dev)
            model.gradient_checkpointing_disable()
        else:
            ids, mask = long_train[idx]
            x, m = torch.tensor([ids], device=dev), torch.tensor([mask], device=dev)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ntok = x.numel()
        t1 = time.time()
        loss, nl = masked_loss(model, x, m)
        (loss * (ntok / a.tokens_per_step)).backward()
        if dev == "mps":
            torch.mps.synchronize()
        res["micro"].append({"kind": kind, "tokens": ntok, "loss": loss.item(), "loss_tokens": nl, "seconds": round(time.time() - t1, 2)})
        if dev == "mps":
            res["micro"][-1]["driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
            res["micro"][-1]["current_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 2)
        del loss
        if dev == "mps":
            drv = res["micro"][-1]["driver_gb"]  # the guard looks at the peak, before the cache is handed back
            torch.mps.empty_cache()  # the allocator's cached blocks go back to the system: otherwise the driver memory only grows (see the doc comment)
            res["micro"][-1]["driver_after_empty_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
        else:
            drv = 0.0
        swp = swap_used_gb() - swap0
        if (a.memory_limit_gb and drv > a.memory_limit_gb) or (a.swap_growth_gb and swp > a.swap_growth_gb):
            model.save_pretrained(os.path.join(a.out, "abort_step%d" % t_step))
            res["aborted"] = {"driver_gb": drv, "swap_growth_gb": swp, "optimizer_step": t_step, "micro": len(res["micro"])}
            json.dump(res, open(os.path.join(a.out, "train.json"), "w"))
            sys.exit(f"memory guard: driver memory {drv:.1f} GB (limit {a.memory_limit_gb}), swap grew {swp:.1f} GB (limit {a.swap_growth_gb}) at optimizer step {t_step}, micro-step {len(res['micro'])}; adapter saved to abort_step{t_step} (optimizer state is not saved)")
        acc += ntok
        if acc >= a.tokens_per_step:
            t_step += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(t_step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            acc = 0
            tail = [r["loss"] for r in res["micro"][-8:]]
            print(f"step {t_step}/{steps} loss(last 8 micro) {np.mean(tail):.4f} elapsed {time.time() - t0:.0f}s", flush=True)
            if t_step % a.eval_every == 0 or t_step == steps or (a.steps_limit and t_step >= a.steps_limit):
                e = evaluate(model, dev, z, long_eval)
                e["step"] = t_step
                res["evals"].append(e)
                model.save_pretrained(os.path.join(a.out, "step%d" % t_step))
                print("  eval", e, flush=True)
                json.dump(res, open(os.path.join(a.out, "train.json"), "w"))
            if a.steps_limit and t_step >= a.steps_limit:
                break
    if not res["evals"] or res["evals"][-1]["step"] != t_step:
        # the planned step count is an estimate from the token total; the loop ends when the stream does, so the last
        # steps were never saved or evaluated (run 3 trained 236 steps of a planned 332 and kept only step 200)
        e = evaluate(model, dev, z, long_eval)
        e["step"] = t_step
        res["evals"].append(e)
        model.save_pretrained(os.path.join(a.out, "step%d" % t_step))
        print("  final eval", e, flush=True)
    res["seconds"] = time.time() - t0
    res["driver_memory_gb"] = torch.mps.driver_allocated_memory() / 1e9 if dev == "mps" else None
    json.dump(res, open(os.path.join(a.out, "train.json"), "w"))


if __name__ == "__main__":
    main()
