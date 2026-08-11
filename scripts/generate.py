#!/usr/bin/env python3
"""Headless video generation against the ComfyUI HTTP API.

Loads an API-format workflow template, rewrites its inputs (prompt, duration,
seed, resolution), submits it to ``POST /prompt`` and polls
``GET /history/{prompt_id}`` until the job finishes.

Stdlib only — no dependencies beyond Python 3.

Nodes are located by ``class_type``, not by node id, so the script keeps
working if the template is re-exported with different ids:

- ``MiniMaxH3ImageToVideo``  -> ``inputs.prompt``
- ``RandomNoise``            -> ``inputs.noise_seed``
- ``ComfyMathExpression``    -> the ``PrimitiveFloat`` feeding it gets the
  duration in seconds; the 17k+5 frame quantisation stays server-side
- ``ResolutionSelector``     -> ``inputs.megapixels`` / ``inputs.aspect_ratio``
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The accelerated template (SageAttention patch + EasyCache, measured ~2.1x at
# equal quality) is the default; pass --template to run the unaccelerated
# baseline in templates/video_minimax_h3_t2v_headless.json for A/B comparisons.
DEFAULT_TEMPLATE = (REPO_ROOT / "templates"
                    / "video_minimax_h3_t2v_headless_accel.json")
DEFAULT_SERVER = "http://localhost:8188"
POLL_INTERVAL_S = 5
PROGRESS_EVERY_S = 60


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


def apply_parameters(workflow, prompt, duration_s, seed,
                     megapixels=None, aspect_ratio=None):
    """Rewrite template inputs in place and return the workflow.

    Args:
        workflow: API-format workflow dict.
        prompt: video prompt text.
        duration_s: clip length in seconds; the ComfyMathExpression node keeps
            enforcing the 17k+5 frame grid server-side.
        seed: noise seed for RandomNoise.
        megapixels: optional ResolutionSelector megapixels override.
        aspect_ratio: optional ResolutionSelector aspect ratio override.
    """
    _, mmx = find_single(workflow, "MiniMaxH3ImageToVideo")
    mmx["inputs"]["prompt"] = prompt

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


def run(prompt, duration_s, seed=None, server=DEFAULT_SERVER,
        template=DEFAULT_TEMPLATE, megapixels=None, aspect_ratio=None,
        timeout_s=3600, wait=True):
    """Submit one generation job and (optionally) wait for its outputs.

    Returns:
        (prompt_id, output_paths). output_paths is empty when wait=False.
    """
    workflow = json.loads(Path(template).read_text())
    if seed is None:
        seed = random.randrange(2**48)
    apply_parameters(workflow, prompt, duration_s, seed,
                     megapixels, aspect_ratio)

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
    parser.add_argument("--duration", type=float, default=5.0,
                        help="clip length in seconds (default: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="noise seed (default: random)")
    parser.add_argument("--megapixels", type=float, default=None,
                        help="ResolutionSelector megapixels (default: template)")
    parser.add_argument("--aspect", default=None,
                        help="ResolutionSelector aspect ratio (default: template)")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"ComfyUI base URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                        help="API-format workflow template")
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

    if args.dry_run:
        workflow = json.loads(Path(args.template).read_text())
        seed = args.seed if args.seed is not None else random.randrange(2**48)
        apply_parameters(workflow, prompt, args.duration, seed,
                         args.megapixels, args.aspect)
        json.dump(workflow, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return

    run(prompt, args.duration, seed=args.seed, server=args.server,
        template=args.template, megapixels=args.megapixels,
        aspect_ratio=args.aspect, timeout_s=args.timeout,
        wait=not args.no_wait)


if __name__ == "__main__":
    main()
