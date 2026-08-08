# minimax-h3-quant-docker

Run MiniMax-H3's **quantised** weights on a **single 16 GB consumer GPU**, in a
Docker container that leaves the host Python environment untouched.

Verified here up to **864×480, 736 frames (30.7 s at 24 fps), with synchronised
audio**, on one RTX 5080 — against an upstream deployment that assumes four GPUs.

`make setup && make models && make up` — that is the whole thing.

## Why this exists

The upstream MiniMax-H3 checkpoints are unrunnable here: the bf16 transformer
alone is 66.3 GB and the Qwen3-VL-32B text encoder another 66.7 GB, and the
official SGLang deployment assumes `--num-gpus 4 --ulysses-degree 4`.

The Comfy-Org repackage changes the arithmetic:

| Component | File | Size |
|---|---|---|
| DiT (FL2VA) | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20.97 GB |
| Text encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.69 GB |
| Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` | 5.21 GB |
| Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB |
| | **Total** | **42.5 GB** |

`pruned` means the AdaLN branches are stripped. The MiniMax README notes that
~13B of the 33B parameters sit in AdaLN branches whose modulation outputs can be
precomputed and cached, so they are not needed for inference-only deployment:
66.28 GB − 40.23 GB = 26 GB = 13B parameters at bf16.

ComfyUI loads the text encoder, DiT and VAE sequentially rather than
simultaneously, so peak VRAM stays under 16 GB while the 42.5 GB working set
lives in system RAM and on NVMe.

## Requirements

Verified on this machine:

- RTX 5080 16 GB (Blackwell, sm_120) — NVFP4 has no hardware path before Blackwell
- 93 GB system RAM
- `~/` on NVMe (measured 3.0 GB/s) — **do not put the models on a network mount**
- Docker 29.2.0 + Compose v5.0.2, `nvidia` runtime registered
- 623 GB free disk

## Layout

```text
minimax-h3-quant-docker/
├── .env.example          template; `make` copies it to .env (gitignored)
├── docker/
│   ├── Dockerfile        CUDA 13.0 + torch 2.12.0+cu130 + SageAttention
│   └── compose.yaml
├── scripts/
│   ├── download_models.sh
│   └── doctor.py         stack verification, run by `make doctor`
├── ComfyUI/              git checkout, bind-mounted to /opt/ComfyUI
└── data/                 bind-mounted to /data (--base-directory)
    ├── models/{diffusion_models,text_encoders,vae}
    ├── custom_nodes/ComfyUI-KJNodes
    ├── input/  output/  user/  temp/
```

Neither `ComfyUI/` nor the contents of `data/` are in version control — the
first is an upstream checkout, the second is 42.5 GB of weights plus runtime
state. Only the empty `data/` skeleton is tracked, so that Compose finds the
bind-mount source already present and owned by you rather than creating it as
root.

## Usage

```bash
make setup     # create .env, clone ComfyUI + KJNodes, build the image
make models    # download 42.5 GB (resumable)
make doctor    # verify GPU, torch, sm_120, SageAttention inside the container
make up        # http://localhost:8188
make logs
make down
```

## Configuration

`.env` is not in version control: it carries this machine's uid/gid and
whatever tuning flags are currently in flight. Any `make` target that talks to
Compose creates it from `.env.example` on first use, substituting your own
`id -u` / `id -g` — the container user is built from those, and a mismatch
leaves everything written into `data/` unwritable. Edit `.env` afterwards;
`.env.example` only supplies the defaults.

| Variable | Default | Purpose |
|---|---|---|
| `COMFYUI_REF` | `v0.30.2` | ComfyUI tag to check out and to pin the image's requirements against |
| `TORCH_VERSION` | `2.12.0` | Installed from the cu130 index |
| `HOST_UID` / `HOST_GID` | your `id -u` / `id -g` | Ownership of bind-mounted files |
| `COMFY_PORT` | `8188` | Host-side port |
| `COMFY_EXTRA_ARGS` | *(empty)* | Appended verbatim to the ComfyUI command line; see [Bring-up order](#bring-up-order) |

## Pinned versions

| | Version | Why |
|---|---|---|
| ComfyUI | `v0.30.2` | `comfy_extras/nodes_minimax_h3.py` and `nodes_easycache.py` are native from 0.30.0 |
| PyTorch | `2.12.0+cu130` | Matches the validated 16 GB Blackwell reference stack; the host's 2.10+cu128 is untested for the NVFP4 dequant path |
| CUDA base | `13.0.1-cudnn-devel` | `devel` so Triton/JIT kernels can build for sm_120 |
| SageAttention | `1.0.6` | Triton-based, no nvcc build step |

## Runtime flags

Set in `docker/compose.yaml`:

- `--base-directory /data` — relocates models, custom_nodes, input, output, temp, user
- `--database-url sqlite:////data/user/comfyui.db` — **does not** follow `--base-directory`
- `--disable-pinned-memory` — measured 45.4 GiB → 6.1 GiB system RAM, no speed penalty
- `--fast-disk` — prefers disk-backed offload over unpinned RAM; worthwhile at 3.0 GB/s

`--use-sage-attention` is deliberately **not** set. A code-path omission in the
MiniMax implementation can produce noise output when SageAttention is forced
globally. Use the KJNodes node inside the workflow instead.

## Nodes that matter on 16 GB

Native to ComfyUI:

- `EasyCache` — skips diffusion steps adaptively

From ComfyUI-KJNodes:

- `MiniMax H3 Memory Efficient Sage Attention Patch`
- `MiniMax H3 Chunk FeedForward` — chunks the SwiGLU FFN over the packed token dim
- `MiniMax H3 Low VRAM Attention` — chunks attention over heads

The SageAttention patch combined with `EasyCache` is reported at ~2.46× speedup.
**None of these are wired into the workflow used for the runs below** — those are
the unaccelerated baseline, so the headroom is still on the table.

## Frame counts are quantised to 17k + 5

The video VAE compresses 17 frames per latent, so a request only lands on a whole
number of latents at `17k + 5` frames — 56, 124, 243, 362, 481, 736. The official
template hides this behind a `ComfyMathExpression` that rounds a duration in
seconds up to the next valid count:

```text
max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17
```

Drive the duration through that expression rather than typing a frame count into
`MiniMaxH3ImageToVideo` directly.

## Performance

### Measured here

21 generations on the reference machine below, all 864×480 (0.4 MP, 16:9,
multiple of 32) at 24 fps with audio, `res_multistep` / `simple` / 20 steps, and
**no acceleration nodes**:

| Frames | Duration | Wall-clock upper bound | Implied |
|---|---|---|---|
| 124 | 5.2 s | ≤ 7.5 min | 3.6 s/frame |
| 243 | 10.1 s | ≤ 15.0 min | 3.7 s/frame |
| 362 | 15.1 s | ≤ 28.1 min | 4.7 s/frame |
| 481 | 20.0 s | ≤ 25.7 min | 3.2 s/frame |
| 736 | 30.7 s | ≤ 46.9 min | 3.8 s/frame |

These are **upper bounds**, not measurements: they are the shortest interval
between consecutive output files of the same configuration, so each still
includes whatever time was spent editing the prompt in between. The consistency
of the implied per-frame cost (3.2–4.7 s) suggests the idle component is small
and that cost scales roughly linearly with frame count, but the authoritative
figures — `Prompt executed in ... seconds` and the VRAM peak — have not been
captured yet.

### Reference figures

From comparable 16 GB Blackwell hardware, for the acceleration this repo has not
yet enabled:

| Job | Time |
|---|---|
| 960×540, 5 s, 20 steps | ~182 s |
| 640×480, 736 frames, 20 steps | ~10 min (optimised) vs ~31 min (not) |

Peak VRAM reported at 12.2–15.3 GiB. Scaling that unoptimised 31 min by the
864×480 / 640×480 pixel ratio gives ~42 min, which the 46.9 min bound above is
consistent with.

## Bring-up order

Start from the official 0.4 MP / 5 s / 20 step template and confirm a stable
baseline before changing anything. Then move one variable at a time — resolution,
then duration, then step count. Changing several at once makes OOM causes
impossible to attribute.

If it OOMs, escalate via `COMFY_EXTRA_ARGS` in `.env`, cheapest first:
`--reserve-vram 1.5`, then `--cache-none`, then `--novram`.

## Verified stack

`make doctor` output on this machine:

```text
python           3.12.3
torch            2.12.0+cu130
cuda build       13.0
device           NVIDIA GeForce RTX 5080
capability       sm_120
vram             15.47 GiB
bf16             True
sageattention    1.0.6
triton           3.7.0
minimax nodes    OK
easycache        OK
ldm.minimax      OK
```

## Troubleshooting

**`Failed to initialize database ... unable to open database file`.** `cli_args.py`
derives the sqlite path from its own `__file__`, not from `--base-directory`, so
it lands on `/opt/ComfyUI/user/comfyui.db` — a directory the git checkout does
not contain. `compose.yaml` passes `--database-url` explicitly to put it in
`/data/user/` with the rest of the state.

**Downloaded fewer files than requested, with exit code 0.** `hf download` is a
Typer CLI with the signature `hf download [OPTIONS] REPO_ID [FILENAMES]...`, and
`--include` takes one glob per occurrence. Passing a list to a single `--include`
routes the remainder into `FILENAMES`, prints "Ignoring `--include` since
filenames have been explicitly set", and silently drops files. `download_models.sh`
passes filenames positionally and verifies every one landed before exiting.

**`Could not resolve 'archive.ubuntu.com'` during build.** The host resolves that
name to AAAA records while the default Docker bridge has no IPv6 route, and
apt's parallel fetches can overrun the LAN resolver. `/etc/apt/apt.conf.d/99-resilient-fetch`
in the image sets `Acquire::Retries "5"` and `Acquire::ForceIPv4 "true"` to cover
both. If it still fails, just re-run `make build` -- earlier layers are cached.

## References

- <https://github.com/Tomiigo/minimax-h3-16gb> — measured 16 GB notes
- <https://huggingface.co/Comfy-Org/MiniMax-H3>
- <https://docs.comfy.org/tutorials/video/minimax/minimax-h3>
- <https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui>
