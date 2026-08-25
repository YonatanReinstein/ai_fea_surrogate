"""Builds the topology-optimization progress video (and stills) embedded in
Kreisselmeier-Steinhauser.pptx/.pdf slide 1, from the real CCX-verified run
in optimization/screenshots_ccx/ (60 generations, hollow_cube).

Each source frame (5120x2880) is mostly dead white space -- title text top
left, then a big gap, then the mesh + colorbar, which itself only occupies
the middle third of the frame's width. crop_frame() below splices out the
title/plot gap (rows 60:745) *and* tightens the horizontal crop to just the
mesh + colorbar (cols 1200:4950 instead of the full 0:4930), so the
geometry fills much more of the video frame. The title-text row band is
cropped to the same final width (0:3750) so it can still be vstacked onto
the now-narrower plot band without resampling either one.

Outputs (optimization/artifacts/):
  topopt_evolution.mp4    -- pptx: embedded, autoplaying video
  topopt_poster.png       -- pptx: poster frame shown before play (last gen)
  topopt_before_after.png -- pdf: static gen-0 vs gen-59 side by side
"""
import glob
import numpy as np
import imageio.v2 as imageio
from PIL import Image

SRC_DIR = "optimization/screenshots_ccx"
OUT_DIR = "optimization/artifacts"
OUT_W = 1280
FPS = 8
HOLD_SECONDS = 2.0

TITLE_ROWS = (0, 60)
TITLE_X = (0, 3750)
PLOT_ROWS = (745, 2840)
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
    out_path = f"{OUT_DIR}/topopt_evolution.mp4"
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
    out_path = f"{OUT_DIR}/topopt_poster.png"
    Image.fromarray(frames[-1]).save(out_path)
    print("wrote", out_path)


def build_before_after(frames):
    before, after = frames[0], frames[-1]
    canvas = Image.fromarray(np.hstack([before, after]))
    half = canvas.resize((canvas.width // 2, canvas.height // 2), Image.LANCZOS)
    out_path = f"{OUT_DIR}/topopt_before_after.jpg"
    half.convert("RGB").save(out_path, quality=87)
    print("wrote", out_path)


def main():
    paths = frame_paths()
    assert paths, f"no frames found in {SRC_DIR}"
    frames = [resized(crop_frame(p)) for p in paths]
    build_video(frames)
    build_poster(frames)
    build_before_after(frames)


if __name__ == "__main__":
    main()
