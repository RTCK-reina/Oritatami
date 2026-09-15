"""Draw the Oritatami app icon (a folded β-meander on a rounded square) with Pillow.

    .venv/bin/python scripts/make_icon.py OUT_DIR

Writes OUT_DIR/icon-1024.png and, on macOS, OUT_DIR/Oritatami.icns (via iconutil).
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
SS = 2  # draw at 2x and downsample for smooth edges


def _meander(n_strand: int = 120, n_turn: int = 90) -> list[tuple[float, float]]:
    """A three-stranded β-meander: up, hairpin, down, hairpin, up — a chain folding back on itself."""
    pts: list[tuple[float, float]] = []

    def strand(x: float, y0: float, y1: float) -> None:
        for i in range(n_strand):
            pts.append((x, y0 + (y1 - y0) * i / (n_strand - 1)))

    def turn(cx: float, cy: float, r: float, a0: float, a1: float) -> None:
        for i in range(n_turn):
            a = a0 + (a1 - a0) * i / (n_turn - 1)
            pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))

    strand(0.315, 0.735, 0.355)
    turn(0.415, 0.355, 0.10, math.pi, 0.0)
    strand(0.515, 0.355, 0.645)
    turn(0.615, 0.645, 0.10, math.pi, 2 * math.pi)
    strand(0.715, 0.645, 0.335)
    return pts


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw() -> Image.Image:
    S = SIZE * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # macOS icon grid: ~824 px rounded square centred in 1024
    box = (100 * SS, 100 * SS, 924 * SS, 924 * SS)
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((box[0], box[1] + 18 * SS, box[2], box[3] + 18 * SS), radius=185 * SS, fill=(0, 0, 0, 120))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18 * SS)))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=185 * SS, fill=255)
    top, bottom = (24, 38, 52), (9, 14, 22)
    grad = Image.new("RGBA", (1, 256))
    for y in range(256):
        grad.putpixel((0, y), _mix(top, bottom, y / 255) + (255,))
    img.paste(grad.resize((S, S)), (0, 0), mask)

    pts = [(x * S, y * S) for x, y in _meander()]
    teal, violet = (88, 213, 201), (167, 139, 250)
    width = 0.078 * S
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i, (x, y) in enumerate(pts[::3]):
        c = _mix(teal, violet, i * 3 / (len(pts) - 1))
        gd.ellipse((x - width, y - width, x + width, y + width), fill=c + (70,))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(26 * SS)))

    d = ImageDraw.Draw(img)
    r = width / 2
    for i, (x, y) in enumerate(pts):
        c = _mix(teal, violet, i / (len(pts) - 1))
        d.ellipse((x - r, y - r, x + r, y + r), fill=c + (255,))
    # highlight stripe along the ribbon
    hr = r * 0.28
    for i, (x, y) in enumerate(pts):
        c = _mix((210, 250, 246), (225, 215, 255), i / (len(pts) - 1))
        d.ellipse((x - r * 0.35 - hr, y - hr, x - r * 0.35 + hr, y + hr), fill=c + (200,))
    # arrowhead on the last strand (β-strand direction)
    tx, ty = pts[-1]
    d.polygon([(tx - 0.085 * S, ty + 0.01 * S), (tx + 0.085 * S, ty + 0.01 * S), (tx, ty - 0.10 * S)], fill=violet + (255,))
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    img = draw()
    img.save(out / "icon-1024.png")
    if sys.platform == "darwin" and shutil.which("iconutil"):
        with tempfile.TemporaryDirectory() as tmp:
            iconset = Path(tmp) / "Oritatami.iconset"
            iconset.mkdir()
            for size in (16, 32, 128, 256, 512):
                img.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
                img.resize((size * 2, size * 2), Image.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out / "Oritatami.icns")], check=True)
    print(out / "icon-1024.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
