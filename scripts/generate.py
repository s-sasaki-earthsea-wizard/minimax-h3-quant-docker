#!/usr/bin/env python3
"""Headless video generation against the ComfyUI HTTP API.

Loads an API-format workflow template, rewrites its inputs (prompt, duration,
seed, resolution, first frame), submits it to ``POST /prompt`` and polls
``GET /history/{prompt_id}`` until the job finishes.

Text-to-video by default; pass ``--image`` for image-to-video and the initial
frame gets wired into the workflow's ``LoadImage``.

Stdlib only — no dependencies beyond Python 3.

Nodes are located by ``class_type``, not by node id, so the script keeps
working if the template is re-exported with different ids:

- ``MiniMaxH3ImageToVideo``  -> ``inputs.prompt``
- ``RandomNoise``            -> ``inputs.noise_seed``
- ``ComfyMathExpression``    -> the ``PrimitiveFloat`` feeding it gets the
  duration in seconds; the 17k+5 frame quantisation stays server-side
- ``ResolutionSelector``     -> ``inputs.megapixels`` / ``inputs.aspect_ratio``
- ``LoadImage``              -> ``inputs.image`` (i2v templates only)
"""

import argparse
import json
import mimetypes
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import image_meta

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
# The accelerated templates (SageAttention patch + EasyCache, measured ~2.1x at
# equal quality) are the defaults; pass --template to run an unaccelerated
# baseline such as templates/video_minimax_h3_t2v_headless.json for A/B
# comparisons. Which of the two applies is decided by --image.
DEFAULT_TEMPLATE_T2V = TEMPLATES_DIR / "video_minimax_h3_t2v_headless_accel.json"
DEFAULT_TEMPLATE_I2V = TEMPLATES_DIR / "video_minimax_h3_i2v_headless_accel.json"
# ComfyUI runs with --base-directory /data and ./data is bind-mounted there, so
# a file already sitting under data/input is addressable by name alone and does
# not need uploading.
COMFY_INPUT_DIR = REPO_ROOT / "data" / "input"
DEFAULT_SERVER = "http://localhost:8188"
POLL_INTERVAL_S = 5
PROGRESS_EVERY_S = 60
# Words that mean the prompt wants somebody to speak. MiniMax-H3 only voices
# what sits literally inside <d>...</d>, so a prompt that merely talks about
# dialogue produces a silent clip -- a trap worth a warning (issue #11).
DIALOGUE_HINTS = ("dialogue", "speaks", "speaking", "says", "saying", "talks",
                  "talking", "conversation", "interview", "monologue",
                  "セリフ", "台詞", "話す", "喋", "発話")


def find_nodes(workflow, class_type):
    """Return [(node_id, node), ...] for every node of the given class_type.

    Args:
        workflow: API-format workflow dict (node_id -> {class_type, inputs}).
        class_type: ComfyUI node class name to look for.
    """
    return [(nid, n) for nid, n in workflow.items()
            if n.get("class_type") == class_type]


def find_single(workflow, class_type):
    """Return the only node of class_type, or exit with a clear error."""
    matches = find_nodes(workflow, class_type)
    if len(matches) != 1:
        sys.exit(f"error: expected exactly one {class_type} node in the "
                 f"template, found {len(matches)}")
    return matches[0]


def resolve_template(template=None, image=None):
    """Return the template path to use, defaulting on whether this is i2v."""
    if template is not None:
        return Path(template)
    return DEFAULT_TEMPLATE_I2V if image is not None else DEFAULT_TEMPLATE_T2V


def upload_image(path, server):
    """POST an image to /upload/image and return its ComfyUI-side name.

    Used for initial frames that live outside data/input, so the script also
    works against a ComfyUI that does not share this checkout's filesystem.
    The server hashes on collision, so re-sending the same file is a no-op
    rather than a "name (1).png" pile-up.
    """
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="image"; filename="'
        + path.name.encode() + b'"\r\n',
        f"Content-Type: {mime}\r\n\r\n".encode(),
        path.read_bytes(), b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="type"\r\n\r\ninput\r\n',
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{server}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            info = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"error: /upload/image returned HTTP {e.code}:\n{body}")
    except urllib.error.URLError as e:
        sys.exit(f"error: ComfyUI unreachable at {server} ({e.reason}); "
                 "is the container up? (make up)")
    subfolder = info.get("subfolder", "")
    name = f"{subfolder}/{info['name']}" if subfolder else info["name"]
    # Informational output goes to stderr so that --dry-run's stdout stays
    # pure JSON.
    print(f"uploaded first frame: {path} -> input/{name}", file=sys.stderr)
    return name


def prepare_image(path, server, upload=True):
    """Return the name a LoadImage node should carry for this file.

    Args:
        path: path to the initial frame on this machine.
        server: ComfyUI base URL, used only when the file has to be uploaded.
        upload: set False for dry runs, which must not touch the server.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        sys.exit(f"error: image not found: {path}")
    try:
        return resolved.relative_to(COMFY_INPUT_DIR).as_posix()
    except ValueError:
        pass
    if not upload:
        print(f"note: {path} is outside data/input and would be uploaded; "
              "the dry run assumes it keeps its name", file=sys.stderr)
        return resolved.name
    return upload_image(resolved, server)


def aspect_for_image(path):
    """Return the ResolutionSelector option matching the image, or None.

    MiniMaxH3ImageToVideo stretches first_frame onto the canvas with the
    upscale method disabled, so a canvas whose aspect disagrees with the image
    distorts the opening frame. Matching it is correctness, not polish.
    """
    size = image_meta.read_size(path)
    if size is None:
        print(f"warning: cannot read the dimensions of {path}; keeping the "
              "template's aspect ratio", file=sys.stderr)
        return None
    ratio = image_meta.nearest_aspect_ratio(*size)
    print(f"first frame: {size[0]}x{size[1]} -> aspect_ratio {ratio!r}",
          file=sys.stderr)
    return ratio


def apply_parameters(workflow, prompt, duration_s, seed,
                     megapixels=None, aspect_ratio=None, image_name=None):
    """Rewrite template inputs in place and return the workflow.

    Args:
        workflow: API-format workflow dict.
        prompt: video prompt text.
        duration_s: clip length in seconds; the ComfyMathExpression node keeps
            enforcing the 17k+5 frame grid server-side.
        seed: noise seed for RandomNoise.
        megapixels: optional ResolutionSelector megapixels override.
        aspect_ratio: optional ResolutionSelector aspect ratio override.
        image_name: initial frame, named as ComfyUI's input directory sees it.
    """
    _, mmx = find_single(workflow, "MiniMaxH3ImageToVideo")
    mmx["inputs"]["prompt"] = prompt

    if image_name is not None:
        if not find_nodes(workflow, "LoadImage"):
            sys.exit("error: an initial frame needs a template with a "
                     "LoadImage node; the t2v templates have none")
        _, loader = find_single(workflow, "LoadImage")
        loader["inputs"]["image"] = image_name

    _, noise = find_single(workflow, "RandomNoise")
    noise["inputs"]["noise_seed"] = seed

    # The duration lives in the PrimitiveFloat referenced by the math node.
    _, math_node = find_single(workflow, "ComfyMathExpression")
    refs = [v for v in math_node["inputs"].values()
            if isinstance(v, list) and len(v) == 2]
    if len(refs) != 1:
        sys.exit("error: ComfyMathExpression should reference exactly one "
                 f"value node, found {len(refs)}")
    duration_node = workflow.get(refs[0][0])
    if duration_node is None or duration_node.get("class_type") != "PrimitiveFloat":
        sys.exit("error: ComfyMathExpression input is not a PrimitiveFloat")
    duration_node["inputs"]["value"] = duration_s

    if megapixels is not None or aspect_ratio is not None:
        _, res = find_single(workflow, "ResolutionSelector")
        if megapixels is not None:
            res["inputs"]["megapixels"] = megapixels
        if aspect_ratio is not None:
            res["inputs"]["aspect_ratio"] = aspect_ratio

    return workflow


def _request_json(url, payload=None, timeout=30):
    """POST payload (or GET when None) and decode the JSON response."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def submit(workflow, server):
    """POST the workflow to /prompt and return the prompt_id."""
    client_id = str(uuid.uuid4())
    try:
        resp = _request_json(f"{server}/prompt",
                             {"prompt": workflow, "client_id": client_id})
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"error: /prompt returned HTTP {e.code}:\n{body}")
    except urllib.error.URLError as e:
        sys.exit(f"error: ComfyUI unreachable at {server} ({e.reason}); "
                 "is the container up? (make up)")
    if resp.get("node_errors"):
        sys.exit("error: workflow rejected:\n"
                 + json.dumps(resp["node_errors"], indent=2))
    return resp["prompt_id"]


def wait_for(prompt_id, server, timeout_s):
    """Poll /history until the job completes; return its history entry."""
    start = time.monotonic()
    last_report = start
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout_s:
            sys.exit(f"error: timed out after {int(elapsed)}s "
                     f"(prompt_id={prompt_id} may still be running)")
        try:
            history = _request_json(f"{server}/history/{prompt_id}")
        except urllib.error.URLError as e:
            sys.exit(f"error: lost contact with {server} ({e.reason})")
        entry = history.get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            return entry
        if entry and entry.get("status", {}).get("status_str") == "error":
            msgs = entry["status"].get("messages", [])
            sys.exit("error: generation failed:\n"
                     + json.dumps(msgs, indent=2))
        if time.monotonic() - last_report >= PROGRESS_EVERY_S:
            print(f"  ... still running ({int(elapsed)}s elapsed)", flush=True)
            last_report = time.monotonic()
        time.sleep(POLL_INTERVAL_S)


def collect_outputs(entry):
    """Return output file paths (relative to data/output/) from a history entry."""
    paths = []
    for node_output in entry.get("outputs", {}).values():
        for value in node_output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and "filename" in item:
                    sub = item.get("subfolder", "")
                    paths.append(f"{sub}/{item['filename']}" if sub
                                 else item["filename"])
    return paths


def warn_if_speech_unwritten(prompt):
    """Warn when a prompt asks for speech but carries no spoken line.

    Asking the model to "include her dialogue in Japanese" is an instruction
    addressed to whoever writes the prompt; the video model reads it as
    description and stays silent.
    """
    if "<d>" in prompt:
        return
    lowered = prompt.lower()
    hint = next((w for w in DIALOGUE_HINTS if w in lowered), None)
    if hint is None:
        return
    print(f"warning: the prompt mentions {hint!r} but carries no "
          "<d>[Japanese]...</d> line, so nothing will be spoken. Write the "
          "line itself, or let the LLM write it: make gen-t2v / gen-i2v "
          "SPEECH=ja", file=sys.stderr)


def run(prompt, duration_s, seed=None, server=DEFAULT_SERVER,
        template=None, megapixels=None, aspect_ratio=None, image=None,
        timeout_s=3600, wait=True):
    """Submit one generation job and (optionally) wait for its outputs.

    Args:
        image: optional initial frame; selects the i2v template unless
            template says otherwise, and sets the canvas aspect ratio unless
            aspect_ratio says otherwise.

    Returns:
        (prompt_id, output_paths). output_paths is empty when wait=False.
    """
    workflow = json.loads(resolve_template(template, image).read_text())
    if seed is None:
        seed = random.randrange(2**48)
    image_name = None
    if image is not None:
        image_name = prepare_image(image, server)
        if aspect_ratio is None:
            aspect_ratio = aspect_for_image(image)
    apply_parameters(workflow, prompt, duration_s, seed,
                     megapixels, aspect_ratio, image_name)

    prompt_id = submit(workflow, server)
    print(f"submitted: prompt_id={prompt_id} seed={seed} "
          f"duration={duration_s}s", flush=True)
    if not wait:
        return prompt_id, []

    entry = wait_for(prompt_id, server, timeout_s)
    outputs = collect_outputs(entry)
    for p in outputs:
        print(f"output: data/output/{p}")
    return prompt_id, outputs


def main():
    parser = argparse.ArgumentParser(
        description="Headless MiniMax-H3 generation via the ComfyUI API")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="video prompt text")
    src.add_argument("--prompt-file",
                     help="read the prompt from a file ('-' for stdin)")
    parser.add_argument("--image", default=None,
                        help="initial frame; switches to the i2v template")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="clip length in seconds (default: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="noise seed (default: random)")
    parser.add_argument("--megapixels", type=float, default=None,
                        help="ResolutionSelector megapixels (default: template)")
    parser.add_argument("--aspect", default=None,
                        help="ResolutionSelector aspect ratio (default: matched "
                             "to --image, else the template's)")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"ComfyUI base URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--template", default=None,
                        help="API-format workflow template (default: the "
                             "accelerated t2v one, or i2v with --image)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="max seconds to wait for completion (default: 3600)")
    parser.add_argument("--no-wait", action="store_true",
                        help="submit and exit without waiting")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the rewritten workflow JSON and exit")
    args = parser.parse_args()

    if args.prompt_file:
        prompt = (sys.stdin.read() if args.prompt_file == "-"
                  else Path(args.prompt_file).read_text())
    else:
        prompt = args.prompt
    # Only on this path: prompts written by pipeline.py are checked there,
    # where the request for speech is known rather than guessed at.
    warn_if_speech_unwritten(prompt)

    if args.dry_run:
        workflow = json.loads(
            resolve_template(args.template, args.image).read_text())
        seed = args.seed if args.seed is not None else random.randrange(2**48)
        image_name, aspect = None, args.aspect
        if args.image is not None:
            image_name = prepare_image(args.image, args.server, upload=False)
            if aspect is None:
                aspect = aspect_for_image(args.image)
        apply_parameters(workflow, prompt, args.duration, seed,
                         args.megapixels, aspect, image_name)
        json.dump(workflow, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return

    run(prompt, args.duration, seed=args.seed, server=args.server,
        template=args.template, megapixels=args.megapixels,
        aspect_ratio=args.aspect, image=args.image, timeout_s=args.timeout,
        wait=not args.no_wait)


if __name__ == "__main__":
    main()
