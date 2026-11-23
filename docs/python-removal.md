# Making cancho-gpu a cancho product: removing the Python (design, 2026-11-17)

> **Status: phase 1 done in this PR (archive of unused examples); phases 2 and 3 are a plan.** Claims are tagged **[measured]**, **[read]** or **[estimate]**. Goal (owner, 2026-11-17): cancho-gpu is a cancho product, like the studio, so it should be mainly cancho. "Helper scripts" (`scripts/`, 12 files, 1.6k lines) are out of scope.

## 1. Inventory **[measured]**, `git ls-files`, main at b629295

| part | files | lines |
|---|---|---|
| all tracked `.py` | 342 | 51,240 |
| `examples/` | 212 | 27,705 (weightspace 14,773, qwen 5,660, cnn 2,551, qwen35 2,375, train 1,021, gpt 1,059, optim 266) |
| `tests/` | 87 | 18,264 |
| `tools/` | 27 | 3,130 |
| the `media` product path (`tools/media.py` and what it runs) | 36 | 7,805 |
| files importing torch, diffusers or transformers | 72 | 14,739 |
| cancho source for scale (`.cho`) | 94 | 38,339 |

## 2. Phase 1: archive what nothing uses (this PR)

**Method [measured]:** a name-based reachability walk (throwaway script, not committed). Roots: `tools/media.py`, every file named in `scripts/gate.sh`, and every tracked non-Python, non-doc file (shell scripts, `.cho`, models, `cases.txt`). Edges: a `*.py` name or an `import` of a tracked module name. Docs and goldens are **not** roots: a document that cites a script does not keep it alive.

**Result:** 195 of 342 Python files are reached; 147 are not, of which **129 are in `examples/` (11,324 lines)**. The other 18 (tests and tools timing scripts, `bench/`) are left for phase 2 because docs cite them as evidence (list in section 4).

**Done:** the 129 files are removed from `examples/`; before that, the tag `archive/python-examples-2026-11-17` was put on main's commit, so every file is one `git show archive/python-examples-2026-11-17:PATH` away. After removal the walk gives the same 195 reached files (nothing needed was cut) and the gate passes (section 5). `examples/` falls from 212 to 83 Python files, 27.7k to 16.4k lines.

**Caveat:** the walk is by name, so it over-keeps (any file sharing a module name with a used one stays) and cannot see a path built at run time. Documents under `docs/` still name some removed scripts as the source of old measurements; they now point at the tag (a sentence in each of the weight-space documents is **not** done here: the claims they make are unchanged, only the place the script lives).

## 3. Phase 2: the orchestration to cancho (plan, **[estimate]**)

`tools/media.py` (the CLI wrapper), the host drivers it calls and the test harnesses that double as drivers (`tests/*_pipeline_gpu.py`, `tests/wan_video_gpu.py`, `tools/wan_text.py`). What they do on the host is file shuffling, process control, numpy arithmetic (samplers, patch embedding, rotary tables, weight conversion) and PNG/MP4 handling. Each of those exists or is small in cancho (`studio/` already runs processes and files; `rewrite_cli.cho` is the pattern). Order: `media` (argument parsing, lock, chaining) first, as `studio/media.cho` calling the same stages; then one driver at a time, each checked against the Python it replaces on the same inputs (byte-equal files where the arithmetic is the same, the existing tolerances where it is not).

## 4. Phase 3: the parts with no cancho equivalent yet (plan)

1. **Text encoders and tokenizers on the host.** umT5 (`tools/wan_text.py`) and the Qwen text/vision encoder and processor run in torch. Removing torch means running them on the GPU in cancho kernels (the Qwen language model already has a GPU path: `tests/text_img_gpu.py`; **umT5 has one since 2026-11-23**, `tests/wan_text_gpu.py`, `docs/diffusion.md` V3: no torch in `media video`, the tokeniser still the `tokenizers` library) plus a tokenizer in cancho. This is new capability, the largest piece.
2. **Image decode/encode (PIL).** PNG read/write and the resize-and-crop in cancho.
3. **Torch oracles.** `tests/` compare against torch; without torch they become recorded goldens (the repo has `golden/`, 11 MB) checked by a cancho program. The cost is losing "regenerate the reference on demand"; docs/diffusion.md already records the torch numbers it cites.
4. **The gate** names 43 Python tests (54 `python3` calls in `scripts/gate.sh`): each is ported or replaced by a golden check.

Candidates left from phase 1 (unreached, cited by docs): `tests/{cuda_emu_sweep,decode_lora,dit_full_gpu,pipeline_gpu,reload,serve_check,time_gemm}.py`, `tests/safetensors/differential.py`, `tools/{diffusion_time,edit_torch_stages,gpu_time,sampler_time,torch_matmul_bench}.py`, `bench/{bench,hf_run,mlx_run}.py`.

## 5. Checks

`scripts/gate.sh` on this branch: pinned compiler a49ea779, `gate: ok`, exit 0. The reachability numbers above are from the script run before and after the removal.

## 6. Rule from here

No new Python in the repository (owner's rule); each phase 2 or 3 PR states the Python lines removed and the cancho lines added, measured.
