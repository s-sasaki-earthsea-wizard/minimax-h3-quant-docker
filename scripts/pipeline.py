#!/usr/bin/env python3
"""LLM-driven headless pipeline: theme -> Ollama -> MiniMax-H3 video.

Chains prompt_gen.py (Ollama, unloads itself with keep_alive: 0) and
generate.py (ComfyUI API). Strictly sequential, so the LLM and the video
model never contend for VRAM. The generated prompts are saved next to the
video outputs for side-by-side review.

Stdlib only — no dependencies beyond Python 3.
"""

import argparse
import json
import random
import time
from pathlib import Path

import generate
import prompt_gen

PROMPTS_DIR = generate.REPO_ROOT / "data" / "output" / "prompts"


def save_prompts(theme, result, model, duration_s, seed):
    """Write the generated prompts (plus metadata) for later review."""
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = PROMPTS_DIR / f"{stamp}_prompts.json"
    path.write_text(json.dumps({
        "theme": theme,
        "model": model,
        "duration_s": duration_s,
        "seed": seed,
        **result,
    }, indent=2, ensure_ascii=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Theme -> Ollama prompts -> MiniMax-H3 video, headless")
    parser.add_argument("theme", help="theme / idea for the clip")
    parser.add_argument("--model", default=prompt_gen.DEFAULT_MODEL,
                        help=f"Ollama model (default: {prompt_gen.DEFAULT_MODEL})")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="clip length in seconds (default: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="noise seed (default: random)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="LLM sampling temperature (default: model default)")
    parser.add_argument("--ollama-server", default=prompt_gen.DEFAULT_SERVER,
                        help=f"Ollama base URL (default: {prompt_gen.DEFAULT_SERVER})")
    parser.add_argument("--comfy-server", default=generate.DEFAULT_SERVER,
                        help=f"ComfyUI base URL (default: {generate.DEFAULT_SERVER})")
    parser.add_argument("--template", default=str(generate.DEFAULT_TEMPLATE),
                        help="API-format workflow template")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="max seconds to wait for the video (default: 3600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="generate and print the prompts, skip the video")
    args = parser.parse_args()

    # Fix the seed up front so the prompts sidecar records the real value.
    seed = args.seed if args.seed is not None else random.randrange(2**48)

    print(f"[1/2] generating prompts with {args.model} ...", flush=True)
    t0 = time.monotonic()
    result = prompt_gen.generate_prompts(
        args.theme, args.duration, model=args.model,
        server=args.ollama_server, temperature=args.temperature)
    print(f"      done in {time.monotonic() - t0:.1f}s")
    print("\n--- image_prompt (unused until i2v) ---")
    print(result["image_prompt"])
    print("\n--- video_prompt ---")
    print(result["video_prompt"])
    print()

    prompts_path = save_prompts(args.theme, result, args.model,
                                args.duration, seed)
    print(f"prompts saved: {prompts_path.relative_to(generate.REPO_ROOT)}")

    if args.dry_run:
        print("dry run: skipping video generation")
        return

    print(f"\n[2/2] generating video ({args.duration}s) ...", flush=True)
    generate.run(result["video_prompt"], args.duration, seed=seed,
                 server=args.comfy_server, template=args.template,
                 timeout_s=args.timeout)


if __name__ == "__main__":
    main()
