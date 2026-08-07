# minimax-h3-quant-docker

Run MiniMax-H3's **quantised** weights on a **single 16 GB consumer GPU**, in a
Docker container that leaves the host Python environment untouched.

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

The SageAttention patch combined with `EasyCache` was measured at ~2.46× speedup.

## Expected performance

Reference figures from comparable 16 GB Blackwell hardware:

| Job | Time |
|---|---|
| 960×540, 5 s, 20 steps | ~182 s |
| 640×480, 736 frames, 20 steps | ~10 min (optimised) vs ~31 min (not) |

Peak VRAM measured at 12.2–15.3 GiB.

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
