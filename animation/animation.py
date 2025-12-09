import numpy as np
import json
import importlib
import os
import cv2
from multiprocessing import Pool
import shutil

FRAME_RATE = 30
NUM_SAMPLES_PER_GEN = 2
NUM_GEN = 3  # (not really used here, but keeping for context)
GEOMETRY = "arm"
N_TURNS = 1  # how many full 360° rotations of the camera


# ============================================================
# Helper functions
# ============================================================

def fill_in_dims(dims1: np.ndarray, dims2: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate between two dimension vectors over FRAME_RATE frames,
    then hold the final one for another FRAME_RATE frames.
    """
    transform = np.linspace(dims1, dims2, FRAME_RATE, axis=0)
    still = np.tile(dims2, (FRAME_RATE, 1))
    return np.vstack((transform, still))


def build_camera_paths(n_frames: int, n_turns: float = 1.0):
    """
    Build smooth elevation and azimuth arrays for all frames.

    - Elevation: gentle sinusoidal variation
    - Azimuth: continuous rotation over n_turns * 360 degrees
      (no modulo, so no wrapping jumps)
    """
    i = np.arange(n_frames, dtype=np.float64)

    # Elevation similar in spirit to your original get_elavation
    elevation = -30.0 + 30.0 * np.sin(np.pi * 0.25 * i / 180.0)

    # Azimuth: start at -180 and rotate continuously
    # from -180 to -180 + 360 * n_turns (exclusive at the end)
    azimuth = -180.0 + (360.0 * n_turns) * (i / n_frames)

    return elevation, azimuth


def vector_to_dict(dim_names, vec):
    return {name: float(vec[i]) for i, name in enumerate(dim_names)}


def make_video_from_frames(folder, output_path, fps=FRAME_RATE):
    frames = sorted(f for f in os.listdir(folder) if f.endswith(".png"))

    if not frames:
        raise ValueError("No PNG images found!")

    first = cv2.imread(os.path.join(folder, frames[0]))
    height, width, _ = first.shape

    video = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    for f in frames:
        frame = cv2.imread(os.path.join(folder, f))
        if frame is None:
            print(f"Warning: could not read {f}, skipping.")
            continue
        video.write(frame)

    video.release()
    print("Video saved to:", output_path)


# ============================================================
# WORKER FUNCTION (runs in each process)
# ============================================================

def render_frame(args):
    """
    Worker: builds model, generates mesh, saves PNG.

    Each worker writes directly to animation/frames/frame_XXXX.png
    (unique filename per frame, so no collisions).
    """
    (i,
     frame_vec,
     dim_names,
     model_path,
     young,
     poisson,
     anchor_condition,
     force_pattern,
     mesh_res,
     frame_gen,
     elev,
     azim) = args

    from core.component import Component
    from core.IritModel import IIritModel

    # Unique file for this frame
    save_path = os.path.join("animation", "frames", f"frame_{i:04d}.png")

    U, V, W = mesh_res
    dims_dict = vector_to_dict(dim_names, frame_vec)

    cad_model = IIritModel(model_path, dims_dict=dims_dict)
    comp = Component(cad_model, young, poisson)

    comp.generate_mesh(U=U, V=V, W=W)
    comp.mesh.anchor_nodes_by_condition(anchor_condition)
    comp.mesh.apply_force_by_pattern(force_pattern)

    comp.mesh.plot_mesh(
        save_path=save_path,
        elev=float(elev),
        azim=float(azim),
        banner=f"Gen {frame_gen}",
    )

    return i  # just return index for logging


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Clean old frames dir
    if os.path.exists("animation/frames"):
        shutil.rmtree("animation/frames")
    os.makedirs("animation/frames", exist_ok=True)

    base_path = f"data/{GEOMETRY}"
    model_path = f"{base_path}/CAD_model/model.irt"

    with open(f"{base_path}/CAD_model/dims.json", "r") as f:
        dims_template = json.load(f)

    with open(f"{base_path}/CAD_model/material_properties.json", "r") as f:
        material_props = json.load(f)
        young = material_props["young_modulus"]
        poisson = material_props["poisson_ratio"]

    bc_module = importlib.import_module(f"data.{GEOMETRY}.boundary_conditions")
    anchor_condition = bc_module.anchor_condition
    force_pattern = bc_module.force_pattern
    mesh_resolution = bc_module.mesh_resolution
    U, V, W = mesh_resolution()

    dim_names = list(dims_template.keys())

    # --------------------------------------------------------
    # Build the list of GA samples
    # --------------------------------------------------------
    samples = []
    samples_gens = []
    gens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 288 ,300, 384]

    for gen_idx in gens:
        gen_pop = np.load(f"optimization/artifacts_V2/ga_population_gen_{gen_idx}.npy")

        # NOTE: assumes gen_pop[0], gen_pop[1], ... are already in the desired order
        for j in range(NUM_SAMPLES_PER_GEN):
            samples.append(gen_pop[j])
            samples_gens.append(gen_idx)

    # --------------------------------------------------------
    # Interpolate between consecutive samples to build frames
    # --------------------------------------------------------
    frames_list = []
    frames_gen = []

    for i in range(len(samples) - 1):
        fill = fill_in_dims(samples[i], samples[i + 1])  # (2*FRAME_RATE, dim)
        frames_list.append(fill)
        frames_gen.extend([samples_gens[i]] * fill.shape[0])

    frames = np.vstack(frames_list)  # shape: (N_frames, dim)
    frames_gen = np.array(frames_gen)
    n_frames = frames.shape[0]

    print(frames.shape)
    print(frames_gen.shape)

    print(f"Total frames: {n_frames}")

    # --------------------------------------------------------
    # Build camera paths for all frames
    # --------------------------------------------------------
    elevations, azimuths = build_camera_paths(n_frames, n_turns=N_TURNS)

    # --------------------------------------------------------
    # Build argument list for multiprocessing
    # --------------------------------------------------------
    tasks = [
        (
            i,
            frames[i],
            dim_names,
            model_path,
            young,
            poisson,
            anchor_condition,
            force_pattern,
            (U, V, W),
            int(frames_gen[i]),
            elevations[i],
            azimuths[i],
        )
        for i in range(n_frames)
    ]

    # --------------------------------------------------------
    # RUN MULTIPROCESSING
    # --------------------------------------------------------
    print("Starting multiprocessing...")
    # Limit processes so IRIT/MAPDL/etc. don't choke on too many parallel calls
    results = []
    with Pool(processes=5) as pool:
        for idx in pool.imap_unordered(render_frame, tasks):
            print(f"Rendered frame {idx}")
            results.append(idx)

    # Optionally verify all frames were rendered
    missing = [
        i for i in range(n_frames)
        if not os.path.exists(os.path.join("animation", "frames", f"frame_{i:04d}.png"))
    ]
    if missing:
        print("Warning: missing frames:", missing)

    # --------------------------------------------------------
    # MAKE VIDEO
    # --------------------------------------------------------
    make_video_from_frames(
        folder="animation/frames",
        output_path="animation/movie.mp4",
        fps=FRAME_RATE,
    )
