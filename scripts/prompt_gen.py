#!/usr/bin/env python3
"""Generate MiniMax-H3 prompts from a theme with a local Ollama model.

Two modes, one function:

- t2v: one chat call produces BOTH an image prompt and a video prompt, so the
  opening frame and the motion stay consistent.
- i2v: the opening frame already exists, so its description is handed *to* the
  model and only the video prompt comes back.

The video prompt follows MiniMax's own prompt format: three labelled fields
(``integrated_multimodal_description`` / ``overall_soundscape`` /
``non_diegetic_music``), preceded in i2v by a fixed first-frame instruction
line. The LLM writes the three fields as separate JSON values and this module
assembles the final text, so the labels, their order and the instruction line
cannot drift.

Output is constrained with Ollama's JSON-schema structured output, and
``keep_alive: 0`` unloads the model from VRAM immediately after the response —
the ComfyUI job that follows gets the whole GPU.

Source for the format: the ``h3-prompt-writing`` skill shipped in the MiniMax-H3
repository (``skills/h3-prompt-writing/references/base-en.txt``).

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

# The three fields MiniMax-H3 expects, in the order it expects them.
FIELD_ORDER = ("integrated_multimodal_description", "overall_soundscape",
               "non_diegetic_music")

# ComfyUI presents a first frame to the text encoder as "<Picture 1>: " plus
# the vision block, ahead of the prompt (comfy/text_encoders/minimax.py), so
# the prompt refers to it by that name. This is the instruction line the I2VA
# guide requires as the prompt's first line.
I2V_ALIGNMENT = ("For the target video, at 0.00 seconds into the target video, "
                 "<Picture 1> (from [Shot 1]) is fully referenced.")

# t2v: both prompts in one response so the initial frame and the video agree.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "image_prompt": {"type": "string"},
        **{name: {"type": "string"} for name in FIELD_ORDER},
    },
    "required": ["image_prompt", *FIELD_ORDER],
}

# i2v: the opening frame is an input, so there is nothing to write for it.
I2V_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "string"} for name in FIELD_ORDER},
    "required": list(FIELD_ORDER),
}

# SPEECH= accepts either an ISO code or the language name; the tag written into
# <d> is always the English name, which is what every official example uses.
LANGUAGE_NAMES = {
    "ja": "Japanese", "jp": "Japanese", "en": "English", "zh": "Chinese",
    "cn": "Chinese", "ko": "Korean", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
}
NO_SPEECH_VALUES = {"none", "no", "off", "silent", "silence"}

_INTRO = """\
You are a prompt writer for MiniMax-H3, a video generation model that renders
video together with synchronised stereo audio: voice, sound effects and music
are generated jointly with the picture in a single pass, not added afterwards.

MiniMax-H3 has its own prompt format, and it is not free-form. Follow the field
contract below exactly.
"""

# The three core fields, shared by both modes.
_FIELDS = """\
- "integrated_multimodal_description": the main body. One continuous
  plain-text block that walks the timeline — visual style, composition,
  subjects, environment, actions, camera moves, who speaks and what they say,
  and the diegetic sound those actions make. Every detail must be something a
  camera can see or a microphone can hear.
- "overall_soundscape": 1-4 sentences covering ambience, action sounds and
  non-verbal human sounds across the whole clip (wind, traffic, footsteps,
  fabric, breathing, laughter). Never repeat dialogue or music here. Use "N/A"
  only for a deliberately silent clip.
- "non_diegetic_music": 1-3 sentences on score the characters cannot hear —
  instrumentation, tempo, rhythm, dynamics. No mood words, no explanation of
  what the music conveys. Music a character can hear (a radio, a busker) is a
  diegetic event and belongs in the description instead. Use "N/A" when there
  is no score.
"""

_SHOTS = """\
Shots, inside "integrated_multimodal_description":
- Open with "[Shot 1] " followed by the visual style and the initial
  composition: "[Shot 1] Live-action, cinematic, a medium-wide shot frames ...".
  Shot 1 never carries a timestamp.
- Every later shot opens with its cut time, strictly increasing and inside the
  clip: "[Shot 2] At 00:03.500, the camera cuts to ...".
- Prefer few shots. A cut has to introduce new information about the subject,
  the space or the moment; when only the framing changes, move the camera
  instead of cutting.
- Camera moves are written as English actions, not stacked labels: zoom in /
  zoom out, push in, pull out, pan left / right, truck left / right, tilt up /
  down, pedestal up / down, arc shot, tracking shot, static shot, shake
  slightly / strongly, POV, roll clockwise / counterclockwise — optionally
  "with small / large amplitude" and "at slow / fast speed":
    The camera pushes in with small amplitude at slow speed toward her hands.
- Text that is really visible on screen goes in double quotes, verbatim and
  untranslated: A red neon sign reading "営業中" glows above the doorway.
"""

_DIALOGUE = """\
Speech:
- Speakers get stable ids (S1), (S2), reused across shots; (S1,S2) when they
  speak at the same time. Characters who never make a sound get no id.
- When a speaker first appears, establish the voice: age, gender, pitch,
  timbre, pace or accent.
- The speaker's description, id, verb and colon stay OUTSIDE the tag. Inside
  <d> goes the language tag and the spoken line itself, nothing else:

    The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>
    The two children (S1,S2) shout together: <d>[English] Wait for us!</d>

- WRITE THE LINE, NEVER DESCRIBE IT. "She speaks about the war in Japanese"
  produces no speech at all. The words that have to be heard must appear
  literally inside <d>, in the language they are spoken in:

    The veteran pilot, hoarse and unhurried (S1), answers: <d>[Japanese]あの冬のことは、今でも夢に見るの。</d>

- The language tag is the language name in English — [Japanese], [English],
  [Chinese] — and it leads the line. Never translate or reword a line you were
  given; copy it verbatim, punctuation included.
- A voiceover uses the exact phrase "says in an off-screen voiceover", and the
  sentence right after </d> states that the character's lips stay closed.
- Keep a line short enough to be spoken inside its shot: roughly 7 Japanese
  characters, or 2-3 English words, per second.
"""

_RULES = """\
Rules:
- Write everything in English except the lines inside <d> and on-screen text.
- The action must fit the given duration, and every cut time stays inside it.
- Be concrete and visual: physical action, materials, light, sound sources.
- Never mention frame counts, resolutions, seeds or model settings.
- No plot summary, no backstory, nothing the camera and microphone cannot
  capture.
"""

_T2V_EXAMPLE = """\
Worked example (5 seconds, two speakers):

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames two mechanics in a lamplit garage at night, a motorcycle on a paddock stand between them. The camera pushes in with small amplitude at slow speed as the older mechanic, gravelly and unhurried (S1), wipes his hands on a rag and says: <d>[Japanese]このキャブ、まだ生きてるな。</d> [Shot 2] At 00:03.000, the shot cuts to a close-up of the younger mechanic's oil-stained hands on the throttle as she answers, bright and quick (S2): <d>[Japanese]じゃあ、朝までに直しましょう。</d>

overall_soundscape: Low night ambience with the hum of a work lamp and distant traffic outside the open shutter. A wrench clicks against metal, a rag rustles, and the throttle cable snaps taut.

non_diegetic_music: N/A
"""

_I2V_EXAMPLE = """\
Worked example (5 seconds, from an illustrated first frame):

integrated_multimodal_description: [Shot 1] Detailed anime illustration style, a medium shot of the young woman shown in <Picture 1>, standing indoors near a doorway. Preserve her face, bun hairstyle, dark dress layered over a light turtleneck, tights and overall appearance from <Picture 1>, along with the doorframe behind her and the warm afternoon light. The camera holds a static shot with a subtle handheld feel as she touches the side of her hair, leans a little closer, blinks once, and, gentle and slightly teasing (S1), says: <d>[Japanese]ねえ、これから一緒に出かけない？</d> Her pose, proportions and identity stay consistent throughout the shot.

overall_soundscape: Quiet indoor ambience with faint clothing rustle and soft breathing under clear, close speech.

non_diegetic_music: N/A
"""

SYSTEM_PROMPT = f"""\
{_INTRO}
Given a theme and a clip duration, respond with JSON containing these fields:

- "image_prompt": a Stable Diffusion style still-image prompt for the opening
  frame of the clip (subject, composition, lighting, style keywords, one
  paragraph). It must depict the same scene the video starts on.
{_FIELDS}
{_SHOTS}
{_DIALOGUE}
{_RULES}
{_T2V_EXAMPLE}"""

I2V_SYSTEM_PROMPT = f"""\
{_INTRO}
The opening frame of this clip has already been rendered and is fed to the
model as a fixed image, presented to it as <Picture 1>. A description of that
frame is given below the theme. Your job is the motion that grows out of it.

Respond with JSON containing these fields:
{_FIELDS}
Rules for the fixed opening frame:
- Shot 1 opens on it. Name the subject as the one "shown in <Picture 1>", then
  write an explicit preservation clause listing what must not change — face,
  hair, wardrobe, key props, the room, the light: "Preserve her face, ... and
  overall appearance from <Picture 1>."
- Close the shot by restating that pose, proportions and identity stay
  consistent. Saying it twice is deliberate, not redundant.
- Carry the frame's art style into the opening style words. An illustrated
  frame means an animated illustration, not live action; only call for live
  action when the frame is photographic.
- Nothing later in the clip may contradict the frame. Anything that changes
  has to be something the clip shows happening.
- The description may be the comma-separated tag list an image generator was
  driven with. Read it as prose, and ignore quality boilerplate ("masterpiece",
  "best quality"), weighting syntax such as parentheses or colons, and anything
  that is not visible in the picture.

{_SHOTS}
{_DIALOGUE}
{_RULES}
{_I2V_EXAMPLE}"""

_SPEECH_REQUIRED = """\

This clip MUST contain spoken dialogue in {language}. At least one speaker
says at least one line, written out as literal {language} text inside
<d>[{language}] ... </d>. Describing that somebody speaks is a failure of the
task: the words themselves have to be in the prompt, and they have to be in
{language}, not in English.
"""

_SPEECH_NONE = """\

This clip has no spoken dialogue at all. Do not use <d> tags or speaker ids,
and do not describe anybody talking.
"""


def resolve_speech(value):
    """Normalise a SPEECH= value into the language to demand.

    Args:
        value: an ISO code ("ja"), a language name ("Japanese"), one of
            NO_SPEECH_VALUES, or None.

    Returns:
        The English name of the language when speech is required, "" when it
        is explicitly ruled out, or None when the value was empty — in which
        case the model decides from the theme.
    """
    if value is None or not value.strip():
        return None
    key = value.strip()
    if key.lower() in NO_SPEECH_VALUES:
        return ""
    return LANGUAGE_NAMES.get(key.lower(), key[:1].upper() + key[1:])


def speech_directive(language):
    """Return the instruction block for a resolved speech language, or ""."""
    if language is None:
        return ""
    if language == "":
        return _SPEECH_NONE
    return _SPEECH_REQUIRED.format(language=language)


def build_video_prompt(fields, first_frame=False):
    """Assemble the final MiniMax-H3 prompt from the three generated fields.

    Args:
        fields: dict carrying every name in FIELD_ORDER.
        first_frame: True when a still is wired into the workflow, which adds
            the fixed <Picture 1> instruction line ahead of the fields.
    """
    blocks = [f"{name}: {fields[name].strip()}" for name in FIELD_ORDER]
    if first_frame:
        blocks.insert(0, I2V_ALIGNMENT)
    return "\n\n".join(blocks)


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


def _chat(payload, server, schema):
    """POST one /api/chat request and return the parsed, validated fields."""
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


def generate_prompts(theme, duration_s, model=DEFAULT_MODEL,
                     server=DEFAULT_SERVER, temperature=None,
                     image_prompt=None, speech=None, first_frame=None):
    """Ask Ollama for a video prompt, and for an image prompt in t2v mode.

    Args:
        theme: user theme / idea for the clip.
        duration_s: clip length in seconds, embedded in the request so the
            cut times fit inside the clip.
        model: Ollama model name.
        server: Ollama base URL.
        temperature: optional sampling temperature.
        image_prompt: description of an existing opening frame. When given,
            the call switches to i2v prompting.
        speech: resolved speech language (see resolve_speech): a language name
            to require, "" to rule speech out, or None to leave it open.
        first_frame: whether a still is wired into the workflow, which decides
            the <Picture 1> instruction line. Defaults to "whenever a frame
            description was supplied".

    Returns:
        dict with the three MiniMax-H3 fields plus the assembled
        "video_prompt", and "image_prompt" in t2v mode.
    """
    if first_frame is None:
        first_frame = image_prompt is not None
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
    if speech is not None:
        system += speech_directive(speech)
        user_lines.append(
            f"Spoken dialogue: {'none' if speech == '' else speech}")

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

    # A prompt that only *talks about* speaking is exactly how a clip ends up
    # silent, so an explicit language request is verified rather than trusted:
    # one more LLM call is cheap next to a wasted GPU run.
    for attempt in range(2):
        result = _chat(payload, server, schema)
        if not speech or "<d>" in result["integrated_multimodal_description"]:
            break
        print(f"warning: {model} wrote no <d> line although {speech} speech "
              f"was requested; retrying ({attempt + 1}/2)", file=sys.stderr)
    else:
        sys.exit(f"error: {model} would not write a <d> line in {speech}. "
                 "Try a different model, raise --temperature, or put the line "
                 "you want spoken in the theme itself.")

    result["video_prompt"] = build_video_prompt(result, first_frame=first_frame)
    return result


def add_image_prompt_arguments(parser):
    """Register the three ways of describing an existing opening frame."""
    parser.add_argument("--image-prompt", default=None,
                        help="description of the opening frame; switches to "
                             "i2v prompting")
    parser.add_argument("--image-prompt-file", default=None,
                        help="read that description from a file")


def add_speech_argument(parser):
    """Register --speech, shared by prompt_gen and pipeline."""
    parser.add_argument("--speech", default=None,
                        help="require spoken dialogue in this language "
                             "('ja', 'Japanese', ...), or 'none' to rule "
                             "speech out (default: the model decides)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MiniMax-H3 prompts from a theme via Ollama")
    parser.add_argument("theme", help="theme / idea for the clip")
    parser.add_argument("--image", default=None,
                        help="opening frame to describe from its embedded "
                             "generation metadata (PNG)")
    add_image_prompt_arguments(parser)
    add_speech_argument(parser)
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
                              image_prompt=image_prompt,
                              speech=resolve_speech(args.speech),
                              first_frame=bool(args.image))
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
