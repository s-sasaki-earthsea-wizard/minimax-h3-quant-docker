"""Verify the container stack can actually run MiniMax-H3 on this GPU.

Run via `make doctor`. Exits non-zero if a hard requirement is missing.
"""

import importlib.metadata
import importlib.util
import sys

FAILURES: list[str] = []
WARNINGS: list[str] = []


def row(label: str, value: object) -> None:
    print(f"  {label:<16} {value}")


def dist_version(name: str) -> str | None:
    """Return an installed distribution's version, or None if absent.

    Package `__version__` attributes are unreliable (sageattention has none),
    so read the installed distribution metadata instead.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


print("\n== runtime ==")
row("python", sys.version.split()[0])

import torch  # noqa: E402  (imported after the header so the banner prints first)

row("torch", torch.__version__)
row("cuda build", torch.version.cuda)

if not torch.version.cuda or not torch.version.cuda.startswith("13"):
    WARNINGS.append(
        f"torch is built against CUDA {torch.version.cuda}, not 13.x. "
        "The NVFP4 text encoder path is only validated on cu130."
    )

print("\n== gpu ==")
if not torch.cuda.is_available():
    FAILURES.append("torch.cuda.is_available() is False -- no GPU visible in the container")
    row("available", False)
else:
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    vram_gib = props.total_memory / 1024**3

    row("device", torch.cuda.get_device_name(0))
    row("capability", f"sm_{major}{minor}")
    row("vram", f"{vram_gib:.2f} GiB")
    row("bf16", torch.cuda.is_bf16_supported())

    # NVFP4 has no hardware path before Blackwell (sm_120). The Comfy-Org text
    # encoder is nvfp4_awq, so anything older silently loses this route.
    if major < 12:
        WARNINGS.append(
            f"sm_{major}{minor} predates Blackwell; the nvfp4_awq text encoder "
            "has no hardware path. Use the int8_convrot encoder instead."
        )

    if vram_gib < 15.0:
        WARNINGS.append(
            f"{vram_gib:.2f} GiB VRAM. Measured peak for H3 is 12.2-15.3 GiB, "
            "so this is tight. Expect to need --reserve-vram or --cache-none."
        )

print("\n== acceleration ==")
for pkg, required in (("sageattention", True), ("triton", False)):
    version = dist_version(pkg)
    row(pkg, version or "MISSING")
    if version is None and required:
        FAILURES.append(f"{pkg} is not installed")

print("\n== comfyui ==")
# ComfyUI is bind-mounted, so a stale or missing checkout is a real failure mode.
sys.path.insert(0, "/opt/ComfyUI")
for label, spec in (
    ("minimax nodes", "comfy_extras.nodes_minimax_h3"),
    ("easycache", "comfy_extras.nodes_easycache"),
    ("ldm.minimax", "comfy.ldm.minimax.model"),
):
    found = importlib.util.find_spec(spec) is not None
    row(label, "OK" if found else "MISSING")
    if not found:
        FAILURES.append(f"{spec} not found under /opt/ComfyUI -- run `make checkout`")

print()
for warning in WARNINGS:
    print(f"  WARN  {warning}")
for failure in FAILURES:
    print(f"  FAIL  {failure}")

if FAILURES:
    print(f"\n{len(FAILURES)} blocking issue(s).\n")
    sys.exit(1)

print("\nAll checks passed.\n")
