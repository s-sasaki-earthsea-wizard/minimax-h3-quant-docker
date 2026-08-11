#!/usr/bin/env python3
"""Generate MiniMax-H3 prompts from a theme with a local Ollama model.

Two modes, one function:

- t2v: one chat call produces BOTH an image prompt and a video prompt, so the
  opening frame and the motion stay consistent.
- i2v: the opening frame already exists, so its description is handed *to* the
  model and only the video prompt comes back.

Output is constrained with Ollama's JSON-schema structured output, and
``keep_alive: 0`` unloads the model from VRAM immediately after the response —
the ComfyUI job that follows gets the whole GPU.

Stdlib only — no dependencies beyond Python 3.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import image_meta

DEFAULT_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q4_K_M"
DEFAULT_SERVER = "http://localhost:11434"
LLM_TIMEOUT_S = 600

# t2v: both prompts in one response so the initial frame and the video agree.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "image_prompt": {"type": "string"},
        "video_prompt": {"type": "string"},
    },
    "required": ["image_prompt", "video_prompt"],
}

# i2v: the opening frame is an input, so there is nothing to write for it.
I2V_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "video_prompt": {"type": "string"},
    },
    "required": ["video_prompt"],
}

_INTRO = """\
You are a prompt writer for MiniMax-H3, a video generation model that renders
video together with synchronised stereo audio: voice, sound effects and music
are generated jointly with the picture in a single pass, not added afterwards.
"""

# The shape of the video_prompt field, shared by both modes.
_VIDEO_PROMPT_STRUCTURE = """\
- "video_prompt": the full MiniMax-H3 prompt, as a single plain-text block
  with this structure:

  1. Style line: the overall look (e.g. "Realistic live-action cinematic
     look, practical film photography style, shallow depth of field, ...").
  2. Scene overview: one short paragraph describing the setting and action.
  3. Storyboard: shots with explicit time ranges covering the whole clip:
       [0s-1.5s] Shot 1: <what happens, camera angle>
       [1.5s-3s] Shot 2: ...
  4. Camera: cut style and movement notes (one line).
  5. Audio: ambience, sound effects and music cues tied to the timeline
     (one short paragraph starting with "Audio:").
  6. Final line: "No text, subtitles, logos or watermarks of any kind."
"""

_DIALOGUE_AND_RULES = """\
Optional dialogue: append spoken lines to a shot as (S1) <d>[ja]...</d>,
where S1/S2/... numbers the speaker and [ja]/[en] tags the language of the
line. Only use dialogue when the theme calls for it.

Rules:
- The storyboard must exactly span the given duration; never exceed it.
- Keep every shot at least 1 second long (the model runs at 24 fps).
- Use concrete, visual language: physical actions, materials, light.
- Write descriptions in English. Dialogue may be Japanese or English.
- Do not mention frame counts, resolutions or model settings.
"""

SYSTEM_PROMPT = f"""\
{_INTRO}
Given a theme and a clip duration, respond with JSON containing two fields:

- "image_prompt": a Stable Diffusion style still-image prompt for the opening
  frame of the clip (subject, composition, lighting, style keywords, one
  paragraph). It must depict the same scene the video starts on.
{_VIDEO_PROMPT_STRUCTURE}
{_DIALOGUE_AND_RULES}"""

I2V_SYSTEM_PROMPT = f"""\
{_INTRO}
The opening frame of this clip has already been rendered and is fed to the
model as a fixed image. A description of it is given below the theme. Your job
is the motion that grows out of that frame.

Respond with JSON containing one field:

{_VIDEO_PROMPT_STRUCTURE}
{_DIALOGUE_AND_RULES}
Rules for the fixed opening frame:
- Shot 1 must start exactly on it: same subject, wardrobe, hair, pose,
  framing, location and lighting. Describe that frame, do not redesign it.
- Nothing later in the clip may contradict it. Anything that changes has to be
  something the clip shows happening.
- Carry its art style into the style line. An illustrated frame means an
  animated illustration, not live action; only call for live action when the
  frame is photographic.
- The description may be the comma-separated tag list an image generator was
  driven with. Read it as prose, and ignore quality boilerplate
  ("masterpiece", "best quality"), weighting syntax such as parentheses or
  colons, and anything that is not visible in the picture.
"""


def resolve_image_prompt(image=None, text=None, path=None):
    """Return the description of an already-rendered opening frame, or None.

    Precedence: an explicit string, then a file, then the generation prompt
    the image generator embedded in the PNG's metadata.

    Args:
        image: path to the opening frame, read for embedded metadata.
        text: description supplied verbatim on the command line.
        path: file to read the description from.
    """
    if text:
        return text.strip()
    if path:
        return Path(path).read_text().strip()
    if image:
        return image_meta.read_generation_prompt(image)
    return None


def generate_prompts(theme, duration_s, model=DEFAULT_MODEL,
                     server=DEFAULT_SERVER, temperature=None,
                     image_prompt=None):
    """Ask Ollama for a video prompt, and for an image prompt in t2v mode.

    Args:
        theme: user theme / idea for the clip.
        duration_s: clip length in seconds, embedded in the request so the
            storyboard timestamps fit.
        model: Ollama model name.
        server: Ollama base URL.
        temperature: optional sampling temperature.
        image_prompt: description of an existing opening frame. When given,
            the call switches to i2v mode and returns only "video_prompt".

    Returns:
        dict with "video_prompt", plus "image_prompt" in t2v mode.
    """
    user_lines = [f"Theme: {theme}", f"Clip duration: {duration_s} seconds"]
    if image_prompt:
        system = I2V_SYSTEM_PROMPT
        schema = I2V_RESPONSE_SCHEMA
        user_lines.append(
            "Opening frame (already rendered, cannot be changed):\n"
            f"{image_prompt}")
    else:
        system = SYSTEM_PROMPT
        schema = RESPONSE_SCHEMA

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(user_lines)},
        ],
        "format": schema,
        "stream": False,
        "keep_alive": 0,
    }
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    req = urllib.request.Request(
        f"{server}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        sys.exit(f"error: Ollama returned HTTP {e.code}:\n{detail}")
    except urllib.error.URLError as e:
        sys.exit(f"error: Ollama unreachable at {server} ({e.reason}); "
                 "is 'ollama serve' running?")

    content = body.get("message", {}).get("content", "")
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        sys.exit(f"error: model did not return valid JSON:\n{content}")
    missing = [k for k in schema["required"]
               if not str(result.get(k) or "").strip()]
    if missing:
        sys.exit(f"error: model response is missing {missing}")
    return result


def add_image_prompt_arguments(parser):
    """Register the three ways of describing an existing opening frame."""
    parser.add_argument("--image-prompt", default=None,
                        help="description of the opening frame; switches to "
                             "i2v prompting")
    parser.add_argument("--image-prompt-file", default=None,
                        help="read that description from a file")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MiniMax-H3 prompts from a theme via Ollama")
    parser.add_argument("theme", help="theme / idea for the clip")
    parser.add_argument("--image", default=None,
                        help="opening frame to describe from its embedded "
                             "generation metadata (PNG)")
    add_image_prompt_arguments(parser)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"Ollama base URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="clip length in seconds (default: 5)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature (default: model default)")
    args = parser.parse_args()

    image_prompt = resolve_image_prompt(args.image, args.image_prompt,
                                        args.image_prompt_file)
    if args.image and not image_prompt:
        print(f"warning: no generation metadata in {args.image}; falling back "
              "to t2v prompting", file=sys.stderr)

    result = generate_prompts(args.theme, args.duration, model=args.model,
                              server=args.server,
                              temperature=args.temperature,
                              image_prompt=image_prompt)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
