#!/usr/bin/env python3
"""Generate MiniMax-H3 prompts from a theme with a local Ollama model.

One chat call produces BOTH an image prompt (reserved for the future i2v
stage) and a video prompt, so the two stay consistent. Output is constrained
with Ollama's JSON-schema structured output, and ``keep_alive: 0`` unloads
the model from VRAM immediately after the response — the ComfyUI job that
follows gets the whole GPU.

Stdlib only — no dependencies beyond Python 3.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q4_K_M"
DEFAULT_SERVER = "http://localhost:11434"
LLM_TIMEOUT_S = 600

# Both prompts in one response so the initial frame and the video agree.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "image_prompt": {"type": "string"},
        "video_prompt": {"type": "string"},
    },
    "required": ["image_prompt", "video_prompt"],
}

SYSTEM_PROMPT = """\
You are a prompt writer for MiniMax-H3, a video generation model that renders
video together with synchronised stereo audio: voice, sound effects and music
are generated jointly with the picture in a single pass, not added afterwards.

Given a theme and a clip duration, respond with JSON containing two fields:

- "image_prompt": a Stable Diffusion style still-image prompt for the opening
  frame of the clip (subject, composition, lighting, style keywords, one
  paragraph). It must depict the same scene the video starts on.
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


def generate_prompts(theme, duration_s, model=DEFAULT_MODEL,
                     server=DEFAULT_SERVER, temperature=None):
    """Ask Ollama for an image prompt and a video prompt.

    Args:
        theme: user theme / idea for the clip.
        duration_s: clip length in seconds, embedded in the request so the
            storyboard timestamps fit.
        model: Ollama model name.
        server: Ollama base URL.
        temperature: optional sampling temperature.

    Returns:
        dict with "image_prompt" and "video_prompt".
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Theme: {theme}\nClip duration: {duration_s} seconds"},
        ],
        "format": RESPONSE_SCHEMA,
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
    missing = [k for k in ("image_prompt", "video_prompt")
               if not result.get(k, "").strip()]
    if missing:
        sys.exit(f"error: model response is missing {missing}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate MiniMax-H3 prompts from a theme via Ollama")
    parser.add_argument("theme", help="theme / idea for the clip")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"Ollama base URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="clip length in seconds (default: 5)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature (default: model default)")
    args = parser.parse_args()

    result = generate_prompts(args.theme, args.duration, model=args.model,
                              server=args.server,
                              temperature=args.temperature)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
