"""Builds the GNN-surrogate-driven topology-optimization progress video (and
poster frame) embedded in Kreisselmeier-Steinhauser.pptx's closing slide,
from optimization/screenshots/ (75 generations, hollow_cube, GNNEvaluator-
driven search -- see optimize_hc_topopt_gnn_68378559.log). Each frame's
stress coloring is still a real per-generation CCX re-solve (screenshot.py
heatmap=True), only the optimizer's own search decisions used the GNN
surrogate (evaluators/gnn_evaluator.py) instead of CCX.

Mirrors make_topopt_video.py's crop/resize pipeline; see that file's
docstring for why the crop rows are needed (dead white space between the
title text and the mesh plot in the raw 5120x2880 screenshots).

Outputs (optimization/artifacts/):
  topopt_evolution_gnn.mp4  -- pptx: embedded, autoplaying video
  topopt_poster_gnn.png     -- pptx: poster frame shown before play (last gen)
"""
import glob
import numpy as np
import imageio.v2 as imageio
from PIL import Image

SRC_DIR = "optimization/screenshots"
OUT_DIR = "optimization/artifacts"
OUT_W = 1280
FPS = 8
HOLD_SECONDS = 2.0

# Tight crop around just the mesh + colorbar (cols 1200:4950) instead of the
# full frame width (cols 0:4930) -- the mesh itself only spans cols
# 1284:3836 and the colorbar 4608:4911, everything else in the raw
# 5120x2880 screenshot is dead white space. The title-text row band is
# cropped to the same width (0:3750) so it can still be vstacked onto the
# now-narrower plot band without resampling either one.
TITLE_ROWS = (0, 60)
TITLE_X = (0, 3750)
PLOT_ROWS = (755, 2830)
PLOT_X = (1200, 4950)


def crop_frame(path):
    arr = np.array(Image.open(path).convert("RGB"))
    top = arr[TITLE_ROWS[0]:TITLE_ROWS[1], TITLE_X[0]:TITLE_X[1]]
    bot = arr[PLOT_ROWS[0]:PLOT_ROWS[1], PLOT_X[0]:PLOT_X[1]]
    return np.vstack([top, bot])


def resized(arr, width=OUT_W):
    h, w = arr.shape[:2]
    out_h = int(round(width * h / w / 2) * 2)  # even, for yuv420p
    return np.array(Image.fromarray(arr).resize((width, out_h), Image.LANCZOS))


def frame_paths():
    return sorted(glob.glob(f"{SRC_DIR}/topopt_gen_*.png"))


def build_video(frames):
    out_path = f"{OUT_DIR}/topopt_evolution_gnn.mp4"
    hold_frames = int(HOLD_SECONDS * FPS)
    writer = imageio.get_writer(
        out_path, fps=FPS, codec="libx264", pixelformat="yuv420p",
        output_params=["-crf", "23"],
    )
    for f in frames:
        writer.append_data(f)
    for _ in range(hold_frames):
        writer.append_data(frames[-1])
    writer.close()
    print("wrote", out_path)


def build_poster(frames):
    out_path = f"{OUT_DIR}/topopt_poster_gnn.png"
    Image.fromarray(frames[-1]).save(out_path)
    print("wrote", out_path)


def main():
    paths = frame_paths()
    assert paths, f"no frames found in {SRC_DIR}"
    frames = [resized(crop_frame(p)) for p in paths]
    build_video(frames)
    build_poster(frames)


if __name__ == "__main__":
    main()
