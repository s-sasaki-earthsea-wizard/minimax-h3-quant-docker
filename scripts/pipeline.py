#!/usr/bin/env python3
"""LLM-driven headless pipeline: theme -> Ollama -> MiniMax-H3 video.

Chains prompt_gen.py (Ollama, unloads itself with keep_alive: 0) and
generate.py (ComfyUI API). Strictly sequential, so the LLM and the video
model never contend for VRAM. The generated prompts are saved next to the
video outputs for side-by-side review.

With ``--image`` the run becomes image-to-video: the still is wired in as the
clip's first frame, and the description of it that the LLM writes from comes
from ``--image-prompt``, ``--image-prompt-file`` or — by default — the
generation prompt the image generator embedded in the PNG.

Stdlib only — no dependencies beyond Python 3.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import generate
import prompt_gen

PROMPTS_DIR = generate.REPO_ROOT / "data" / "output" / "prompts"
# An i2v run already carries an idea in the picture, so the theme is optional
# there; this stands in for "just animate what is in the frame".
DEFAULT_I2V_THEME = ("Bring the opening frame to life with subtle, natural "
                     "motion that suits the scene. Do not change the subject, "
                     "the setting or the time of day.")


def save_prompts(theme, result, model, duration_s, seed,
                 image=None, image_prompt=None, speech=None):
    """Write the generated prompts (plus metadata) for later review.

    Args:
        image: path to the first frame in i2v runs.
        image_prompt: the description of that frame handed to the LLM. Kept
            under its own key so it is never confused with the still-image
            prompt the model *writes* in t2v runs.
        speech: the resolved dialogue language the run asked for, if any.
    """
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = PROMPTS_DIR / f"{stamp}_prompts.json"
    path.write_text(json.dumps({
        "theme": theme,
        "model": model,
        "duration_s": duration_s,
        "seed": seed,
        "speech": speech,
        "first_frame": str(image) if image else None,
        "first_frame_prompt": image_prompt,
        **result,
    }, indent=2, ensure_ascii=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Theme -> Ollama prompts -> MiniMax-H3 video, headless")
    parser.add_argument("theme", nargs="?", default=None,
                        help="theme / idea for the clip (optional with --image)")
    parser.add_argument("--image", default=None,
                        help="still to use as the clip's first frame; "
                             "switches the whole run to i2v")
    prompt_gen.add_image_prompt_arguments(parser)
    prompt_gen.add_speech_argument(parser)
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
    parser.add_argument("--template", default=None,
                        help="API-format workflow template (default: the "
                             "accelerated t2v one, or i2v with --image)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="max seconds to wait for the video (default: 3600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="generate and print the prompts, skip the video")
    args = parser.parse_args()

    theme = args.theme
    if theme is None:
        if not args.image:
            parser.error("a theme is required unless --image is given")
        theme = DEFAULT_I2V_THEME

    # Fix the seed up front so the prompts sidecar records the real value.
    seed = args.seed if args.seed is not None else random.randrange(2**48)

    image_prompt = prompt_gen.resolve_image_prompt(
        args.image, args.image_prompt, args.image_prompt_file)
    if args.image and not image_prompt:
        print(f"warning: no generation metadata in {args.image} and no "
              "--image-prompt; it will still be the first frame, but the LLM "
              "writes without seeing it", file=sys.stderr)

    speech = prompt_gen.resolve_speech(args.speech)
    mode = "i2v" if image_prompt else "t2v"
    if speech:
        mode += f", {speech} speech"
    elif speech == "":
        mode += ", no speech"
    print(f"[1/2] generating prompts with {args.model} ({mode}) ...",
          flush=True)
    t0 = time.monotonic()
    result = prompt_gen.generate_prompts(
        theme, args.duration, model=args.model,
        server=args.ollama_server, temperature=args.temperature,
        image_prompt=image_prompt, speech=speech,
        first_frame=bool(args.image))
    print(f"      done in {time.monotonic() - t0:.1f}s")
    if image_prompt:
        print("\n--- first frame (given to the LLM) ---")
        print(image_prompt)
    if "image_prompt" in result:
        print("\n--- image_prompt (t2v only; for your image model) ---")
        print(result["image_prompt"])
    print("\n--- video_prompt ---")
    print(result["video_prompt"])
    print()

    prompts_path = save_prompts(theme, result, args.model,
                                args.duration, seed, image=args.image,
                                image_prompt=image_prompt, speech=speech)
    print(f"prompts saved: {prompts_path.relative_to(generate.REPO_ROOT)}")

    if args.dry_run:
        print("dry run: skipping video generation")
        return

    print(f"\n[2/2] generating video ({args.duration}s) ...", flush=True)
    generate.run(result["video_prompt"], args.duration, seed=seed,
                 server=args.comfy_server, template=args.template,
                 image=args.image, timeout_s=args.timeout)


if __name__ == "__main__":
    main()
