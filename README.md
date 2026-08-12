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
│   ├── doctor.py         stack verification, run by `make doctor`
│   ├── generate.py       headless generation via the ComfyUI API (`make gen`)
│   ├── image_meta.py     size / aspect / embedded prompt of a still, stdlib only
│   ├── prompt_gen.py     theme -> prompts via a local Ollama model
│   └── pipeline.py       theme (+ still) -> Ollama -> MiniMax-H3 (`make gen-t2v` / `gen-i2v`)
├── templates/
│   ├── video_minimax_h3_t2v_headless.json         API-format t2v workflow (baseline)
│   ├── video_minimax_h3_t2v_headless_accel.json   + SageAttention patch + EasyCache (default)
│   ├── video_minimax_h3_i2v_headless.json         same, with a LoadImage -> first_frame
│   └── video_minimax_h3_i2v_headless_accel.json   + SageAttention patch + EasyCache (default)
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

Headless (no browser, server must be up):

```bash
make gen     PROMPT="..." DURATION=5        # your prompt, passed through verbatim
make gen-t2v PROMPT="..." DURATION=5        # your prompt -> Ollama rewrites it -> video
make gen-t2v PROMPT="..." SPEECH=ja         # ... and make somebody speak Japanese in it
make gen-t2v PROMPT="..." DRY_RUN=1         # prompts only, review before spending GPU time

make gen     PROMPT="..." IMAGE=data/input/still.png     # still + your prompt, verbatim
make gen-i2v IMAGE=data/input/still.png                  # still -> Ollama writes the motion -> video
make gen-i2v IMAGE=... PROMPT="..." SPEECH=ja DRY_RUN=1  # steer it, review the prompt first
```

`gen` hands your text to the video model unchanged; `gen-t2v` / `gen-i2v` send
it to a local LLM first, which rewrites it into
[MiniMax-H3's own prompt format](#the-prompt-format) and injects the result.

## Headless generation

The browser UI is only a client: it assembles a workflow JSON and POSTs it to
the server. `scripts/generate.py` (stdlib only) does the same against an
API-format export of the workflow — by default the accelerated
`templates/video_minimax_h3_t2v_headless_accel.json`, or its i2v counterpart
when `--image` is given (both carry the SageAttention patch + EasyCache,
[measured ~2.1×](#measured-here) at visually equal quality). Pass
`--template templates/video_minimax_h3_t2v_headless.json` for the
unaccelerated baseline:

- Nodes are located by `class_type`, not node id, so re-exporting the
  template does not break the script.
- The duration is passed in **seconds**; the `ComfyMathExpression` node stays
  in the workflow, so the [17k + 5 frame quantisation](#frame-counts-are-quantised-to-17k--5)
  is still enforced server-side.
- Jobs POSTed to `/prompt` queue server-side and run sequentially, so batch
  submission needs no extra logic.

### Image to video

`--image path/to/still.png` (`make gen IMAGE=...`) selects the i2v template and
writes the still into its `LoadImage` node. Two details the script settles on
your behalf:

- **Aspect ratio.** `MiniMaxH3ImageToVideo` stretches `first_frame` onto the
  canvas with the upscale method *disabled*, so a canvas that disagrees with
  the still visibly distorts the opening frame. The `ResolutionSelector` aspect
  ratio is therefore matched to the image's own dimensions unless `--aspect`
  overrides it; the megapixel target is left alone.
- **Where the file lives.** `data/` is bind-mounted at `/data` and ComfyUI runs
  with `--base-directory /data`, so a still already under `data/input/` is
  addressed by name alone. Anything elsewhere is POSTed to `/upload/image`
  first, which also lets the script drive a ComfyUI that does not share this
  filesystem.

### LLM-driven prompts

`scripts/pipeline.py` chains a local LLM in front of it: your prompt goes to
Ollama (`make gen-t2v MODEL=...` to pick any `ollama list` entry) in one
schema-constrained call, and what comes back is submitted to ComfyUI. In t2v
mode it returns **both** a still-image prompt and a video prompt, so the two
stay consistent. `make gen-t2v` and `make gen-i2v` are the two entry points;
`make pipeline` is the same thing with the mode taken from whether `IMAGE=` is
set.

In i2v mode the opening frame already exists, so the description travels the
other way — into the LLM. It comes from `--image-prompt`, `--image-prompt-file`
or, by default, the generation prompt the image generator left in the PNG's
`parameters` text chunk, parsed straight out of the file by
`scripts/image_meta.py` (no Pillow). The model is told the frame is fixed, that
shot 1 must open on it, and that the frame's art style rules the clip — an
illustrated still stays an illustration instead of drifting to live action.
With `--image` the theme itself is optional; omit it and the LLM is simply
asked to bring the frame to life.

The request sets `keep_alive: 0`, unloading the LLM from VRAM before ComfyUI
starts; the pipeline is strictly sequential, so the models never contend.
Generated prompts are saved to `data/output/prompts/` next to the videos for
side-by-side review. Requires Ollama running on the host (default
`http://localhost:11434`).

#### The prompt format

MiniMax-H3 does not take free-form text. Its prompt is three labelled fields,
optionally preceded by a keyframe instruction line:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Detailed anime illustration style, a medium shot of the young woman shown in <Picture 1> ... and, gentle and slightly teasing (S1), says: <d>[Japanese]ねえ、これから一緒に出かけない？</d> [Shot 2] At 00:03.500, the camera cuts to ...

overall_soundscape: Quiet indoor ambience with faint clothing rustle and soft breathing.

non_diegetic_music: N/A
```

- The instruction line is only for i2v, and `<Picture 1>` is not decoration:
  ComfyUI presents a first frame to the text encoder as `"<Picture 1>: "` plus
  the vision block, ahead of the prompt (`comfy/text_encoders/minimax.py`), so
  that is the name the prompt has to use for it.
- `[Shot 1]` carries no timestamp; every later shot opens with a strictly
  increasing cut time, `[Shot 2] At 00:03.500, ...`. Camera moves are written
  as English actions with an optional amplitude and speed.
- `overall_soundscape` covers ambience and action sound; `non_diegetic_music`
  covers score the characters cannot hear, and is `N/A` rather than omitted
  when there is none.

The LLM returns the three fields as separate JSON values and `prompt_gen.py`
assembles the text, so the labels, their order and the instruction line cannot
drift — only the prose inside them is generated. The format follows the
`h3-prompt-writing` skill shipped in the MiniMax-H3 repository
(`skills/h3-prompt-writing/references/base-en.txt`).

#### Making somebody speak

MiniMax-H3 voices exactly what sits inside a `<d>` tag, and nothing else.
*Asking* for dialogue in the prompt — "please include her dialogue in
Japanese" — is read as description and produces a silent clip; the words to be
heard have to be written out:

```text
The veteran pilot, hoarse and unhurried (S1), answers: <d>[Japanese]あの冬のことは、今でも夢に見るの。</d>
```

The speaker's description, id, verb and colon stay **outside** the tag; inside
it go the language tag and the line itself, verbatim. Speaker ids `(S1)`,
`(S2)` are stable across shots, `(S1,S2)` for people speaking together. The
language tag is the English name of the language — `[Japanese]`, `[English]`
— and the ISO form `[ja]` also works.

`SPEECH=ja` (also `SPEECH=Japanese`, `en`, ...) makes dialogue in that language
a hard requirement of the LLM call; `SPEECH=none` rules speech out; leaving it
unset lets the model decide from your prompt. Because a prompt that only talks
*about* speaking is the exact failure this flag exists to prevent, the answer
is checked for a `<d>` tag and the call is retried once before giving up — one
extra LLM call is cheap next to a wasted GPU run. `make gen`, which passes your
text through untouched, warns when the prompt asks for speech but carries no
`<d>` line.

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
| `SAGEATTENTION_REF` | `d1a57a5...` | thu-ml/SageAttention commit compiled into the image |
| `HOST_UID` / `HOST_GID` | your `id -u` / `id -g` | Ownership of bind-mounted files |
| `COMFY_PORT` | `8188` | Host-side port |
| `COMFY_EXTRA_ARGS` | *(empty)* | Appended verbatim to the ComfyUI command line; see [Bring-up order](#bring-up-order) |

## Pinned versions

| | Version | Why |
|---|---|---|
| ComfyUI | `v0.30.2` | `comfy_extras/nodes_minimax_h3.py` and `nodes_easycache.py` are native from 0.30.0 |
| PyTorch | `2.12.0+cu130` | Matches the validated 16 GB Blackwell reference stack; the host's 2.10+cu128 is untested for the NVFP4 dequant path |
| CUDA base | `13.0.1-cudnn-devel` | `devel` so nvcc can compile SageAttention and Triton/JIT kernels for sm_120 |
| SageAttention | source @ `d1a57a5` (2.2.0) | The PyPI 1.0.6 release lacks the int8/fp8 kernels the KJNodes MiniMax patch imports — the node registers but fails at execution. Built from source for sm_120 only (~106 s) |

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

The SageAttention patch and `EasyCache` are wired into the default headless
template and [measured here at ~2.1×](#measured-here) on 124 frames, with no
VRAM penalty and no visible quality difference on same-seed pairs. The two
chunking nodes remain unused: peak VRAM is unchanged by the patch, so there is
nothing for them to buy yet.

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

**Acceleration A/B** — same image, same container, same prompt, same seed;
only the template differs. 864×480, 124 frames (5.2 s), 20 steps, audio on.
Times are the server's own `Prompt executed in`; VRAM is a 2 s `nvidia-smi`
poll (upper bound, desktop included):

| Template | Seed | Time | s/frame | Peak VRAM |
|---|---|---|---|---|
| baseline | 42 | 151.1 s | 1.22 | 15.2 GiB |
| accelerated | 42 | **68.3 s** | 0.55 | 15.2 GiB |
| baseline | 43 | 134.6 s | 1.09 | 15.2 GiB |
| accelerated | 43 | **64.5 s** | 0.52 | 14.3 GiB |

The like-for-like pair (both with the text encoding already node-cached) gives
**2.09×**. EasyCache alone contributed 1.43–1.54× (6–7 of 20 steps skipped, at
its default `reuse_threshold` 0.2); the rest is the SageAttention patch.
Same-seed output pairs were compared visually and are indistinguishable. The
~2.46× reference figure was reported at 736 frames, where fixed costs amortise
further, so these shorter-clip numbers are consistent with it.

Two caveats for anyone re-measuring: ComfyUI returns a byte-identical workflow
from its execution cache in 0.00 s, so repeat runs must vary the seed; and a
prompt shared across templates keeps its cached text encoding, which is what
makes the pairs above like-for-like.

**Unaccelerated history** — 21 generations, all 864×480 at 24 fps with audio,
`res_multistep` / `simple` / 20 steps, **no acceleration nodes**:

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

From comparable 16 GB Blackwell hardware:

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
sageattention    2.2.0
triton           3.7.0
kjnodes kernels  OK
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
