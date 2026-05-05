"""
generate_demo_data.py — SARDS Demo Dataset Generator
======================================================
Creates synthetic but realistic Dhaka satellite images for years
2018–2024 WITHOUT requiring a live internet connection or GEE account.

Each generated image realistically simulates:
  - Gradual urban expansion  (increasing bright pixels year-over-year)
  - Declining green cover    (decreasing vegetation channel response)
  - River/water body shifts  (moving blue polygon)
  - Sensor noise             (Gaussian noise layer)
  - Night-light increase     (Perlin-like bright cluster growth)

Run:
    python scripts/generate_demo_data.py

Output:
    data/dhaka/2018.jpg … 2024.jpg

Author : SARDS Team
Version: 1.0.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import numpy as np
import cv2

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/dhaka")
IMG_W, IMG_H = 512, 512
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
RNG = np.random.default_rng(seed=42)


# ── Base scene builder ────────────────────────────────────────────────────────

def make_base_scene(year: int, progression: float) -> np.ndarray:
    """
    Generate a single-year synthetic satellite image.

    Parameters
    ----------
    year        : Calendar year (used only for seeding randomness).
    progression : 0.0 = earliest year, 1.0 = latest year.

    Returns
    -------
    np.ndarray  shape (H, W, 3) RGB uint8
    """
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)

    # ── 1. Rural/peri-urban background: muted green-brown ─────────────
    base_r = 100 + 30  * progression          # Urban = slightly brighter
    base_g = 130 - 45  * progression          # Vegetation declining
    base_b = 80  + 20  * progression
    img[:, :, 0] = base_r
    img[:, :, 1] = base_g
    img[:, :, 2] = base_b

    # ── 2. River / water body — blue polygon, shifting west ─────────
    river_shift = int(30 * progression)       # River moves westward
    river_x1 = max(40,  80  - river_shift)
    river_x2 = max(65,  105 - river_shift)
    img[50:460, river_x1:river_x2, 0] = 30
    img[50:460, river_x1:river_x2, 1] = 90
    img[50:460, river_x1:river_x2, 2] = 180 + int(20 * (1 - progression))
    # River bank encroachment (width shrinks over time)
    river_width = max(10, 25 - int(15 * progression))
    img[50:460, river_x1:river_x1 + river_width, 0] = 35
    img[50:460, river_x1:river_x1 + river_width, 1] = 95
    img[50:460, river_x1:river_x1 + river_width, 2] = 175

    # ── 3. Urban core — bright grey/white expanding block ─────────────
    core_size = int(100 + 80 * progression)
    cx, cy = IMG_W // 2, IMG_H // 2
    x1 = max(0, cx - core_size // 2)
    x2 = min(IMG_W, cx + core_size // 2)
    y1 = max(0, cy - core_size // 2)
    y2 = min(IMG_H, cy + core_size // 2)
    brightness = 160 + int(70 * progression)
    img[y1:y2, x1:x2, :] = brightness

    # Road grid within urban core
    for road_offset in range(-core_size // 2, core_size // 2, 25):
        rx = cx + road_offset
        ry = cy + road_offset
        if 0 <= rx < IMG_W:
            img[y1:y2, max(0, rx-1):min(IMG_W, rx+2), :] = 200
        if 0 <= ry < IMG_H:
            img[max(0, ry-1):min(IMG_H, ry+2), x1:x2, :] = 200

    # ── 4. Peri-urban sprawl patches ──────────────────────────────────
    n_sprawl = int(8 + 18 * progression)
    sprawl_rng = np.random.default_rng(seed=int(year))
    for _ in range(n_sprawl):
        sx = sprawl_rng.integers(0, IMG_W - 30)
        sy = sprawl_rng.integers(0, IMG_H - 30)
        sw = sprawl_rng.integers(12, 35)
        sh = sprawl_rng.integers(12, 35)
        col = 130 + sprawl_rng.integers(0, 80)
        img[sy:sy+sh, sx:sx+sw, :] = col

    # ── 5. Vegetation patches (shrinking) ─────────────────────────────
    n_veg = int(20 - 14 * progression)
    veg_rng = np.random.default_rng(seed=int(year) + 1)
    for _ in range(n_veg):
        vx = veg_rng.integers(0, IMG_W - 40)
        vy = veg_rng.integers(0, IMG_H - 40)
        vw = veg_rng.integers(15, 45)
        vh = veg_rng.integers(15, 45)
        img[vy:vy+vh, vx:vx+vw, 0] = 60
        img[vy:vy+vh, vx:vx+vw, 1] = 130 + int(30 * (1 - progression))
        img[vy:vy+vh, vx:vx+vw, 2] = 55

    # ── 6. Nightlight cluster (south side) ────────────────────────────
    nl_intensity = int(60 * progression)
    if nl_intensity > 0:
        nl_cx = int(IMG_W * 0.65)
        nl_cy = int(IMG_H * 0.70)
        nl_r  = int(20 + 40 * progression)
        for dy in range(-nl_r, nl_r):
            for dx in range(-nl_r, nl_r):
                dist = np.sqrt(dx**2 + dy**2)
                if dist < nl_r:
                    px = nl_cx + dx
                    py = nl_cy + dy
                    if 0 <= px < IMG_W and 0 <= py < IMG_H:
                        fade = max(0, 1 - dist / nl_r)
                        img[py, px, :] = np.clip(
                            img[py, px, :] + nl_intensity * fade, 0, 255
                        )

    # ── 7. Gaussian noise (sensor simulation) ─────────────────────────
    noise = RNG.normal(0, 6, (IMG_H, IMG_W, 3)).astype(np.float32)
    img = np.clip(img + noise, 0, 255)

    return img.astype(np.uint8)


# ── NDVI-like green tint helper ───────────────────────────────────────────────

def add_seasonal_variation(img: np.ndarray, year: int) -> np.ndarray:
    """
    Apply slight seasonal hue shift to simulate dry/wet season imagery.
    Odd years lean slightly greener (wet season); even years warmer (dry).
    """
    out = img.astype(np.float32).copy()
    if year % 2 == 0:  # Dry season — slightly warmer/browner
        out[:, :, 0] = np.clip(out[:, :, 0] * 1.03, 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] * 0.97, 0, 255)
    else:              # Wet season — slightly greener
        out[:, :, 1] = np.clip(out[:, :, 1] * 1.04, 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * 1.02, 0, 255)
    return out.astype(np.uint8)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(YEARS)

    print(f"\n🛰  Generating {n} synthetic Dhaka satellite images …\n")

    for idx, year in enumerate(YEARS):
        progression = idx / max(n - 1, 1)    # 0.0 → 1.0
        img = make_base_scene(year, progression)
        img = add_seasonal_variation(img, year)

        out_path = output_dir / f"{year}.jpg"
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        bar_len = int(progression * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  [{bar}] {year} → {out_path}  ({img.shape})")

    print(f"\n✔  All images saved to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic Dhaka satellite images for SARDS demo."
    )
    parser.add_argument(
        "--output", "-o", default=str(OUTPUT_DIR),
        help="Output directory (default: data/dhaka)"
    )
    args = parser.parse_args()
    generate_all(Path(args.output))
