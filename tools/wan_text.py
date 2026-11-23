#!/usr/bin/env python3
"""The text of a Wan2.2 prompt, on the **host** (torch): umT5-XXL's encoder. Since V3 of docs/diffusion.md section 6 the stage of `media video` is tests/wan_text_gpu.py (the encoder as a GPU session); this is kept as torch's side of its checks.

    wan_text.py "PROMPT" OUT.npy [--device mps|cpu]

Writes the encoder's last hidden state of the prompt's tokens, [n, 4096] float32 (`WanPipeline`'s `_get_t5_prompt_embeds`: the tokens up to the first padding, the state of each), for tools/wan_ref.py's
`text_embed` and tests/wan_video_gpu.py, which pad it with zero rows to 512 (the pipelines do the same before the condition embedder's projection). Needs torch and transformers (the venv of the first image edits)
and `text_encoder/*` and `tokenizer/*` of `Wan-AI/Wan2.2-TI2V-5B-Diffusers` in the Hugging Face cache.
"""
import glob
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

prompt, out = sys.argv[1], sys.argv[2]
dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else ("mps" if torch.backends.mps.is_available() else "cpu")
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/*"))[0]
tok = AutoTokenizer.from_pretrained(snap + "/tokenizer")
enc = UMT5EncoderModel.from_pretrained(snap + "/text_encoder", torch_dtype=torch.bfloat16).to(dev).eval()
t = tok([prompt], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
n = int(t.attention_mask.sum())
with torch.no_grad():
    h = enc(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state[0, :n].float().cpu().numpy()
np.save(out, h)
print("prompt of %d tokens -> %s %s, rms %.4g" % (n, out, list(h.shape), float(np.sqrt((h ** 2).mean()))))
