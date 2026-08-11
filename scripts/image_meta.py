#!/usr/bin/env python3
"""Read what the i2v pipeline needs out of a still image, stdlib only.

Three things, none of which justify pulling in Pillow:

- ``read_size`` — pixel dimensions, straight out of the PNG/JPEG header.
- ``nearest_aspect_ratio`` — the closest ``ResolutionSelector`` option.
  ``MiniMaxH3ImageToVideo`` stretches ``first_frame`` onto the canvas with the
  upscale method disabled, so a canvas whose aspect disagrees with the image
  distorts the opening frame. Matching the ratio is not cosmetic.
- ``read_generation_prompt`` — the positive prompt an image generator left in
  the PNG metadata, used as the "this is what the opening frame shows"
  description handed to the LLM.
"""

import math
import re
import struct
import zlib
from pathlib import Path

# Mirrors comfy_extras/nodes_resolution.py; keys are the literal combo values
# the ResolutionSelector node accepts, values are width / height.
ASPECT_RATIOS = {
    "1:1 (Square)": 1 / 1,
    "2:3 (Portrait Photo)": 2 / 3,
    "3:2 (Photo)": 3 / 2,
    "3:4 (Portrait Standard)": 3 / 4,
    "4:3 (Standard)": 4 / 3,
    "9:16 (Portrait Widescreen)": 9 / 16,
    "16:9 (Widescreen)": 16 / 9,
    "21:9 (Ultrawide)": 21 / 9,
}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# A1111 / Forge write one tEXt chunk keyed "parameters"; ComfyUI writes
# "prompt" and "workflow" instead, which are workflow JSON and not a prompt.
SD_PARAMETERS_KEY = b"parameters"
# The trailer A1111 appends after the prompts: "Steps: 5, Sampler: Euler a, ..."
SD_TRAILER_RE = re.compile(r"^(Steps|Negative prompt):\s", re.IGNORECASE)
LORA_TAG_RE = re.compile(r"<[a-z_]+:[^>]*>", re.IGNORECASE)


def read_size(path):
    """Return (width, height) for a PNG or JPEG, or None if unrecognised.

    Args:
        path: path to the image file.
    """
    data = Path(path).read_bytes()
    if data.startswith(PNG_SIGNATURE):
        # IHDR is mandated to be the first chunk: width and height are the
        # first two big-endian uint32 of its payload, at a fixed offset.
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if data.startswith(b"\xff\xd8"):
        return _jpeg_size(data)
    return None


def _jpeg_size(data):
    """Walk JPEG segments to the frame header and return (width, height)."""
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        # SOF0..SOF15 carry the frame geometry; SOF4/8/12 are not frame headers.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + length
    return None


def nearest_aspect_ratio(width, height):
    """Return the ResolutionSelector option closest to width / height.

    Compared in log space so that, say, 3:2 and 2:3 are judged by the same
    relative error rather than by the difference of the raw quotients.
    """
    target = width / height
    return min(ASPECT_RATIOS,
               key=lambda name: abs(math.log(ASPECT_RATIOS[name] / target)))


def read_generation_prompt(path):
    """Return the positive prompt embedded by the image generator, or None.

    Reads the PNG "parameters" text chunk written by A1111 / Forge, keeps
    everything above the "Negative prompt:" / "Steps:" trailer and drops
    ``<lora:...>``-style tags, which mean nothing outside that sampler.

    Args:
        path: path to the image file.
    """
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        return None
    raw = _png_text_chunk(data, SD_PARAMETERS_KEY)
    if raw is None:
        return None
    prompt = _strip_sampler_trailer(raw)
    return prompt or None


def _png_text_chunk(data, key):
    """Return the decoded text of the tEXt/zTXt/iTXt chunk named key, or None."""
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        offset += 12 + length  # 4 length + 4 type + payload + 4 CRC
        if chunk_type == b"IEND":
            break
        if chunk_type not in (b"tEXt", b"zTXt", b"iTXt"):
            continue
        keyword, _, rest = payload.partition(b"\x00")
        if keyword != key:
            continue
        try:
            if chunk_type == b"tEXt":
                return rest.decode("utf-8", "replace")
            if chunk_type == b"zTXt":
                # rest = compression method byte + zlib stream
                return zlib.decompress(rest[1:]).decode("utf-8", "replace")
            # iTXt: compression flag, compression method, language tag,
            # translated keyword, then the text itself.
            flag = rest[0]
            body = rest[2:].split(b"\x00", 2)[-1]
            if flag:
                body = zlib.decompress(body)
            return body.decode("utf-8", "replace")
        except (zlib.error, IndexError):
            return None
    return None


def _strip_sampler_trailer(raw):
    """Keep the positive prompt, drop the sampler settings and LoRA tags."""
    kept = []
    for line in raw.splitlines():
        if SD_TRAILER_RE.match(line.strip()):
            break
        kept.append(line)
    text = LORA_TAG_RE.sub("", "\n".join(kept))
    # Tag soup reads better to the LLM as one comma-separated line than as the
    # blank-line-separated blocks the image UI encourages.
    parts = [p.strip() for p in text.replace("\n", ",").split(",")]
    return ", ".join(p for p in parts if p)


def main():
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Dump the size, aspect ratio and embedded prompt of an image")
    parser.add_argument("image", help="path to a PNG or JPEG")
    args = parser.parse_args()

    size = read_size(args.image)
    json.dump({
        "width": size[0] if size else None,
        "height": size[1] if size else None,
        "aspect_ratio": nearest_aspect_ratio(*size) if size else None,
        "generation_prompt": read_generation_prompt(args.image),
    }, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
