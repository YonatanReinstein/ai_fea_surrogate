"""Design-of-experiments sampler for tile-thickness datasets: scrambled-Sobol
anchors, spatially correlated thickness fields, and members-on-background
two-level designs, with a cached uniform-thickness volume lookup
(data/<geometry>/dataset/volume_lookup.npz). Used by
datasets/build_cross4_dsig_dataset.py. (Moved 2026-11-23 from the deleted
dataset_builder.py / build_tile_load_dataset.py; see git fed57d9.)
"""
import importlib
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter

from core.IritModel import IritCModel


def _sample_two_level(tile_keys, lo, hi, NX, NY, NZ, rng, sigma,
                       vol_frac, t_thick, beta, t_thin_frac=0.0, back_texture=0.0):
    """Members-on-a-background design: a correlated field pushed through a soft
    threshold, so the tile population is BIMODAL (a thin background plus a
    minority of thick members) instead of a unimodal blob around one anchor.

    Why this exists. The correlated-field sampler varies thickness BETWEEN
    designs (its anchor is drawn uniform in volume) but only slightly WITHIN
    one: measured over 200 samples of the 30k set, the median per-design
    max/min contrast is 3.5, while every design a topopt run reaches sits at
    ~47 -- floor tiles and 0.30+ members in the SAME design. Raising the field
    amplitude does not fix the shape: it pushes mass toward BOTH bounds, so the
    >=0.30 fraction climbs to 32% (target 6%) while the 0.05-0.12 web shrinks
    to 9% (target 41%). A Gaussian field is unimodal; the designs that matter
    are strongly right-skewed -- mostly thin, a few thick members. That skew is
    what the threshold produces.

    Parameters (all DoE factors, see the Sobol block in sample_designs):
      sigma       correlation length of the underlying field, in tiles
      vol_frac    fraction of tiles above the threshold (the "member" share)
      t_thick     thickness the members approach
      beta        transition sharpness; small = soft/graded, large = crisp.
                  None = no threshold at all: a monotone rank map onto
                  [t_back, t_thick] (see below)
      t_thin_frac background thickness as a fraction of (t_thick - lo)
      back_texture spread of the background, as a fraction of (t_back - lo).
                   A flat background puts every non-member tile at one value;
                   real designs carry a textured thin region (41% of tiles in
                   0.05-0.12 with only 9% at the floor), so the background is
                   given its own correlated variation and its own share of
                   floor-clipped tiles.
    """
    pad = int(np.ceil(4.0 * 12.0))
    dims_shape = (NX, NY, NZ)
    pads = tuple(pad if d > 1 else 0 for d in dims_shape)
    modes = tuple("constant" if d > 1 else "reflect" for d in dims_shape)
    noise = rng.standard_normal(tuple(d + 2 * p for d, p in zip(dims_shape, pads)))
    field = gaussian_filter(noise, sigma=sigma, mode=modes)
    field = field[pads[0]:pads[0] + NX, pads[1]:pads[1] + NY, pads[2]:pads[2] + NZ]
    f = field.reshape(-1)
    f = (f - f.mean()) / max(f.std(), 1e-12)
    # threshold at the quantile that leaves exactly vol_frac of tiles above it,
    # so the member share is a CONTROLLED factor rather than an outcome
    if beta is None:
        # NO THRESHOLD: map the field monotonically onto [0, 1] by RANK, so the
        # design spans exactly [t_back, t_thick] with no level-set step. The
        # thresholded form puts mass at two plateaus and its "members" come out
        # as compact blobs (level sets of a smooth field are blobs), which is
        # visibly not what a real design looks like. The rank map keeps the
        # spatial correlation but makes the thickness distribution uniform
        # between the two ends, and -- because the extremes are hit exactly --
        # it makes per-design contrast t_thick/t_back a DIRECTLY CONTROLLED DoE
        # factor instead of an outcome of amplitude and clipping.
        w = np.empty_like(f)
        w[np.argsort(f)] = np.linspace(0.0, 1.0, f.size)
    else:
        thr = np.quantile(f, 1.0 - vol_frac)
        w = 1.0 / (1.0 + np.exp(-beta * (f - thr)))        # soft membership
    t_back = lo + t_thin_frac * (t_thick - lo)
    if back_texture > 0:
        g = gaussian_filter(rng.standard_normal(tuple(d + 2 * p for d, p in zip(dims_shape, pads))),
                            sigma=max(sigma * 0.5, 0.8), mode=modes)
        g = g[pads[0]:pads[0] + NX, pads[1]:pads[1] + NY, pads[2]:pads[2] + NZ].reshape(-1)
        g = (g - g.mean()) / max(g.std(), 1e-12)
        t_back = t_back + back_texture * (t_back - lo) * g
    values = np.clip(t_back + (t_thick - np.maximum(t_back, lo)) * w, lo, hi)
    return dict(zip(tile_keys, values.tolist()))


def _sample_correlated_field(tile_keys, lo, hi, default, NX, NY, NZ, rng,
                              amplitude_frac_up=0.10, amplitude_frac_down=0.10, sigma=None):
    # Per-tile i.i.d. uniform sampling (the plain path below) produces
    # spatially uncorrelated "salt-and-pepper" designs -- nothing like the
    # smooth, spatially-correlated material fields a real structure (or a
    # topopt trajectory) actually has, so the surrogate never sees that
    # regime during training. This generates a smooth random field over the
    # tile grid instead: white noise blurred by a random correlation length
    # (short -> textured, long -> near-uniform). .reshape(-1) on the
    # (NX, NY, NZ) array is C-order (last axis fastest =
    # ix*NY*NZ + iy*NZ + iz), matching the X-outermost ShellThicknesses
    # layout model_lin.c reads for d1..dN.
    #
    # Deliberately NOT per-sample min-max normalized: that would force every
    # sample -- even a heavily-blurred, nearly-constant one -- back up to
    # fill the full [lo, hi] range, which was the original bug here (every
    # field sample ended up with some near-zero-thickness pocket somewhere,
    # pushing stress far above what a real topopt trajectory ever sees).
    #
    # Also deliberately normalized by sigma's *expected* per-cell std
    # (1/(2*sqrt(pi)*sigma) for a 2D Gaussian blur of unit white noise --
    # confirmed empirically) rather than left as a raw fixed amplitude: raw
    # amplitude decays local contrast AND the spatial-averaging effect
    # together as sigma grows, so the field's spatial MEAN (i.e. the
    # design's overall volume) stays locked near `default` regardless of
    # sigma -- reproducing the exact same near-zero volume diversity the
    # i.i.d. sampler had (see [[project_gnn_field_sampling]]), just with
    # smooth texture instead of salt-and-pepper. Normalizing out sigma's
    # decay first keeps per-cell contrast comparable across all sigma, so
    # what changes with sigma is purely how many *independent* correlated
    # regions the grid contains -- large sigma means only a few big
    # correlated blobs, so their average (i.e. overall volume) swings much
    # more sample-to-sample than at small sigma, where ~840 near-independent
    # cells average tightly back to `default`. That swinging overall volume
    # at large sigma is exactly the coverage this sampler exists to add.
    # sigma is a DoE factor when the caller supplies one (Sobol path); the
    # internal draw is the fallback for the plain independent-uniform path.
    sigma = float(sigma) if sigma is not None else rng.uniform(1.0, 12.0)
    # mode="reflect" on the true (NX, NY, NZ) grid inflates variance near
    # any edge -- up to ~2x the interior value right at a corner, verified
    # empirically for this grid shape across sigma in [1, 12] (see
    # [[project_gnn_anchor_sensitivity_bug]]) -- because reflected padding
    # re-applies kernel weight to mirrored *copies* of already-counted noise
    # instead of fresh independent values, and those copies are perfectly
    # correlated with the original, so their cross-covariance adds to the
    # output variance instead of cancelling the way independent terms would.
    # `expected_std` below is only exact for an infinite (no-boundary)
    # domain, so dividing by it directly would leave edge/corner tiles --
    # including whichever tile happens to sit on a boundary condition --
    # systematically over-amplified and more often clipped to [lo, hi] than
    # interior tiles, distorting the local thickness distribution the
    # surrogate trains on exactly where it matters most.
    #
    # Fix: generate noise on an oversized grid, padded by ~4*sigma_max (the
    # Gaussian kernel's effective reach at scipy's default truncate=4) on
    # each side of every axis with real extent (>1 cell), blur that, then
    # crop back to the true (NX, NY, NZ) region. Every true-grid cell --
    # including corners -- now sits far enough from the padded array's own
    # edge that it's blurred against genuine independent noise on all
    # sides, matching the infinite-domain assumption `expected_std` relies
    # on. Verified empirically (hollow_cube's 40x21x1 grid): corner/edge/
    # interior std all land within ~1% of `expected_std` after this change,
    # vs. up to 2x apart before. Axes with only 1 cell (e.g. hollow_cube's
    # NZ) are left unpadded with mode="reflect": padding a size-1 axis
    # would just repeat independent noise every 2*pad cells for no benefit,
    # and reflect there is already a no-op (the single true cell reflects
    # onto itself, contributing its real variance with no suppression or
    # inflation) -- this generalizes to any geometry's tile_grid() shape,
    # not just hollow_cube's specific dimensions.
    pad = int(np.ceil(4.0 * 12.0))
    dims = (NX, NY, NZ)
    pads = tuple(pad if d > 1 else 0 for d in dims)
    modes = tuple("constant" if d > 1 else "reflect" for d in dims)
    noise = rng.standard_normal(tuple(d + 2 * p for d, p in zip(dims, pads)))
    field = gaussian_filter(noise, sigma=sigma, mode=modes)
    field = field[pads[0]:pads[0] + NX, pads[1]:pads[1] + NY, pads[2]:pads[2] + NZ]
    expected_std = 1.0 / (2.0 * np.sqrt(np.pi) * sigma)
    standardized = field.reshape(-1) / expected_std
    # Asymmetric amplitude: thinning a tile (standardized < 0) drives stress
    # up (and is what actually creates the low-volume/thin-design tail this
    # sampler needs -- see [[project_gnn_field_sampling]]), while thickening
    # it (> 0) has sharply diminishing returns on stress (confirmed
    # empirically -- uniform-baseline stress only drops 93.9->88.5 MPa going
    # from thickness 0.30->0.40) and, with an anchor thickened enough to hit
    # the stress target on its own, mostly just clips at `hi` instead of
    # adding useful diversity. A single symmetric amplitude couples these
    # two very different sensitivities together; splitting them lets the
    # anchor be thickened for the stress target while amplitude_frac_down
    # alone controls how much real thin/low-volume coverage there is.
    amp = np.where(standardized < 0, amplitude_frac_down, amplitude_frac_up)
    values = default + amp * (hi - lo) * standardized
    values = np.clip(values, lo, hi)
    return dict(zip(tile_keys, values.tolist()))


def _build_or_load_volume_inverse(base_path, model_path, fixed_dims, field_tile_keys, lo, hi, n_points=300):
    # field_random_base used to draw its per-sample anchor uniformly in
    # THICKNESS, but thickness->volume is a strongly saturating curve for a
    # fixed-topology geometry like hollow_cube (measured: volume(0.01)=33.3,
    # volume(0.49)=838.7, with the top ~25% of the thickness range
    # contributing well under 10% of that span) -- uniform-in-thickness
    # sampling therefore spends most of its density in the flat/high-volume
    # plateau, which is exactly the "lots of high-volume/low-stress samples"
    # skew this fixes. get_volume() is a CAD-only query (no mesh, no FEA), so
    # sweeping a few hundred uniform-thickness points to trace this curve
    # costs seconds, not the hours a calibration sweep of real CalculiX runs
    # would -- cheap enough to do once and cache, then invert so the anchor
    # is drawn uniform in VOLUME instead: pick a target volume, look up the
    # thickness that produces it.
    lookup_path = f"{base_path}/dataset/volume_lookup.npz"
    if os.path.exists(lookup_path):
        npz = np.load(lookup_path)
        thicknesses, volumes = npz["thicknesses"], npz["volumes"]
    else:
        exe_path = os.path.join(model_path, "model")
        exe_win_path = os.path.join(model_path, "model.exe")
        thicknesses = np.linspace(lo, hi, n_points)
        volumes = np.empty(n_points)
        for i, t in enumerate(thicknesses):
            dims = {k: float(t) for k in field_tile_keys}
            exe = exe_path if os.path.isfile(exe_path) else exe_win_path
            m = IritCModel(exe, dims_dict=dims, fixed_dims=fixed_dims)
            volumes[i] = m.get_volume()
        os.makedirs(os.path.dirname(lookup_path), exist_ok=True)
        np.savez(lookup_path, thicknesses=thicknesses, volumes=volumes)
        print(f"Built thickness->volume lookup ({n_points} pts, cached to {lookup_path})", flush=True)
    # volumes is monotonically non-decreasing in thicknesses (verified for
    # hollow_cube), but genuinely flat -- not just noisy -- at the very top
    # of the range: the design is fully solid there, so volume hits a hard
    # ceiling a few points before hi (measured: last ~6 of 300 points on
    # hollow_cube all landed on the exact same volume). PchipInterpolator
    # requires a strictly increasing x, so collapse any duplicate volumes to
    # their first (thinnest) occurrence before inverting -- that's also the
    # better representative of a saturated plateau for this fix's purpose,
    # since the thinnest design reaching a given volume is the one with the
    # most stress diversity among ties.
    volumes_u, uniq_idx = np.unique(volumes, return_index=True)
    thicknesses_u = thicknesses[uniq_idx]
    inverse = PchipInterpolator(volumes_u, thicknesses_u)
    return float(volumes_u[0]), float(volumes_u[-1]), inverse


def sample_designs(geometry, n, seed, two_level_frac, amp_range, t_min=None):
    """Scrambled Sobol over
    (anchor volume, sigma, amp_up, amp_down, two-level params), correlated
    fields with a per-design anchor drawn uniform in volume, and a share of
    members-on-background two-level designs. t_min raises dims.json's floor
    (and the anchor volume's lower end with it)."""
    from scipy.stats import qmc
    base = f"data/{geometry}"
    tmpl = json.load(open(f"{base}/CAD_model/dims.json"))
    bc = importlib.import_module(f"data.{geometry}.boundary_conditions")
    NX, NY, NZ = bc.tile_grid(None)
    keys = [f"d{i + 1}" for i in range(NX * NY * NZ)]
    assert list(tmpl) == keys, "dims.json must be exactly one thickness per tile, d1..dK"
    lo = np.array([tmpl[k]["min"] for k in keys])
    hi = np.array([tmpl[k]["max"] for k in keys])
    v_lo, v_hi, v_inv = _build_or_load_volume_inverse(
        base, f"{base}/CAD_model", bc.fixed_dims(), keys, lo[0], hi[0])
    if t_min is not None:
        lo, v_lo = np.maximum(lo, t_min), volume_at(base, t_min)

    m = int(np.ceil(np.log2(max(n, 2))))
    sob = qmc.Sobol(d=6, scramble=True, seed=seed).random_base2(m)[:n]
    rng = np.random.default_rng(seed)
    a_lo, a_hi = amp_range
    loguni = lambda u, a, b: float(np.exp(np.log(a) + u * (np.log(b) - np.log(a))))
    designs, kinds = [], []
    for i in range(n):
        u = sob[i]
        if u[4] < two_level_frac:
            beta = None if u[5] < 0.5 else loguni(u[5], 2.0, 60.0)
            d = _sample_two_level(keys, lo, hi, NX, NY, NZ, rng,
                                  1.0 + u[1] * 5.0, 0.05 + u[2] * 0.35,
                                  lo[0] + (0.25 + 0.75 * u[0]) * (hi[0] - lo[0]),
                                  beta, (u[3] ** 2) * 0.45,
                                  back_texture=0.3 + 0.5 * u[4] / max(two_level_frac, 1e-9))
            kinds.append("two_level")
        else:
            t0 = float(v_inv(v_lo + u[0] * (v_hi - v_lo)))
            d = _sample_correlated_field(keys, lo, hi, np.full(len(keys), t0), NX, NY, NZ, rng,
                                         amplitude_frac_up=loguni(u[2], a_lo, a_hi),
                                         amplitude_frac_down=loguni(u[3], a_lo, a_hi),
                                         sigma=1.0 + u[1] * 11.0)
            kinds.append("field")
        designs.append({k: float(v) for k, v in d.items()})
    return designs, kinds


def volume_at(base, t):
    """Uniform-thickness volume at t, from the cached volume_lookup.npz that
    _build_or_load_volume_inverse writes."""
    npz = np.load(f"{base}/dataset/volume_lookup.npz")
    return float(np.interp(t, npz["thicknesses"], npz["volumes"]))
