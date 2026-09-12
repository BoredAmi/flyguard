"""Retinotopic encoder: optic flow field -> T4/T5 firing -> LPLC2 drive.

Task 1 of the architecture (CLAUDE.md "Next, in order"). This is the bridge
between a 2-D optic flow field (eventually from MuJoCo-rendered image pairs;
for now, synthetic flow fields analogous to `flyguard.stimuli`) and real
T4/T5 neurons: `column_assignment.csv` gives each T4/T5 neuron's retinotopic
column position (x, y), and its subtype letter (a/b/c/d) gives its
direction preference. The encoder does NOT need to know where LPLC2's
receptive field sits explicitly -- that pooling is already baked into the
real synaptic wiring extracted by `flyguard.extract` (T4/T5 -> LPLC2, see
`inspect_subnetwork.py`). The encoder's only job is: given a flow field,
compute each individual T4/T5 neuron's firing drive from its own column
position and subtype tuning.

Direction mapping (CLAUDE.md "RESOLVED"): a=front-to-back, b=back-to-front,
c=upward, d=downward.

**Stated assumption, not yet validated (CLAUDE.md "Remaining open item")**:
this module fixes a world/image-plane convention --

    +x column coordinate = front (rostral) of the visual field
    +y column coordinate = dorsal (up)

-- and derives each subtype's preferred flow *vector* from that. Whether
this matches the real fly's coordinate frame, and whether it is mirrored
between the left and right optic lobe, is exactly what the per-hemisphere
looming-vs-translation replication test is for. Do not treat the sign of
these vectors as confirmed until that test passes on both sides -- if only
one hemisphere discriminates correctly, mirror `DIRECTION_UNIT_VECTOR` for
the other rather than the column coordinates themselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flyguard.stimuli import SUBTYPE_DIRECTIONS

# Preferred flow direction, as a unit vector in (x, y) column-coordinate
# space, per direction name. See module docstring for the stated
# front/back/up/down <-> +x/-x/+y/-y convention and its caveat.
DIRECTION_UNIT_VECTOR = {
    "front_to_back": np.array([-1.0, 0.0]),   # front (+x) -> back (-x): flow points -x
    "back_to_front": np.array([1.0, 0.0]),
    "upward": np.array([0.0, 1.0]),
    "downward": np.array([0.0, -1.0]),
}

SUBTYPE_UNIT_VECTOR = {
    letter: DIRECTION_UNIT_VECTOR[direction]
    for letter, direction in SUBTYPE_DIRECTIONS.items()
}


def load_column_assignment(csv_path: Path, side: str | None = None) -> pd.DataFrame:
    """Load T4/T5 retinotopic columns from `column_assignment.csv`.

    Returns a DataFrame with root_id, hemisphere, subtype letter, x, y, and
    the neuron's preferred flow unit vector (pref_vx, pref_vy).
    `side`, if given ("left"/"right"), filters to one hemisphere -- used by
    the per-hemisphere replication test.
    """
    df = pd.read_csv(csv_path)
    t4t5 = df[df["type"].str.match(r"^T[45][a-d]$", na=False)].copy()
    if side is not None:
        t4t5 = t4t5[t4t5["hemisphere"] == side]
    t4t5["subtype"] = t4t5["type"].str[-1]
    vecs = np.stack(t4t5["subtype"].map(SUBTYPE_UNIT_VECTOR).to_numpy())
    t4t5["pref_vx"] = vecs[:, 0]
    t4t5["pref_vy"] = vecs[:, 1]
    return t4t5[["root_id", "hemisphere", "type", "subtype", "x", "y", "pref_vx", "pref_vy"]]


def expanding_flow_2d(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flow vector at each (x, y) column for a uniformly expanding
    (looming) stimulus: radially outward from the column-coordinate origin,
    unit magnitude (the origin itself gets zero flow)."""
    r = np.hypot(x, y)
    r_safe = np.where(r == 0, 1.0, r)
    return x / r_safe, y / r_safe


def translational_flow_2d(x: np.ndarray, direction_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Flow vector at every column for pure self-motion translation: the
    same fixed direction everywhere, unit magnitude."""
    theta = np.radians(direction_deg)
    vx = np.full_like(x, np.cos(theta), dtype=np.float64)
    vy = np.full_like(x, np.sin(theta), dtype=np.float64)
    return vx, vy


def encode_drive(columns: pd.DataFrame, flow_vx: np.ndarray, flow_vy: np.ndarray,
                  magnitude_weighted: bool = False, sat_px: float = 2.0) -> pd.Series:
    """Per-neuron rectified-cosine drive: how well the local flow vector
    aligns with each T4/T5 neuron's own subtype-preferred direction.

    This is the direction-selective, half-wave-rectified tuning real T4/T5
    neurons show (Maisak et al. 2013) -- the same rectified-cosine model
    used for the abstract ring circuit in `flyguard.stimuli.arm_responses`,
    now evaluated per real neuron at its own retinotopic column instead of
    pooled into four arms.

    Returns a Series indexed by root_id, drive in [0, 1].

    `magnitude_weighted` (default False, i.e. every prior result in this
    repo is unchanged) additionally scales each neuron's drive by local flow
    *speed*, saturating at `sat_px` pixels/frame. The default False is
    correct for the synthetic flow fields this encoder was built on, whose
    vectors are unit-magnitude by construction -- there, normalising away a
    magnitude that is always 1.0 costs nothing. It is a much stronger
    assumption on *real* estimated flow, where it discards the difference
    between a wall creeping past and an obstacle rushing at the camera, and
    leaves the pooled drive close to a count of how many columns have any
    measurable flow at all. Real T4/T5 are not binary: firing rate grows
    with contrast and with speed up to a preferred temporal frequency, which
    is what the saturating scale models. `flyguard.validate_arena` measures
    both variants against distance-to-obstacle rather than assuming which
    one carries the signal.
    """
    pref = columns[["pref_vx", "pref_vy"]].to_numpy()
    flow = np.stack([flow_vx, flow_vy], axis=1)
    flow_norm = np.linalg.norm(flow, axis=1, keepdims=True)
    flow_unit = np.divide(flow, flow_norm, out=np.zeros_like(flow), where=flow_norm > 0)
    cos_sim = (pref * flow_unit).sum(axis=1)
    drive = np.clip(cos_sim, 0, None)
    if magnitude_weighted:
        speed = np.clip(flow_norm[:, 0] / max(sat_px, 1e-9), 0.0, 1.0)
        drive = drive * speed
    return pd.Series(drive, index=columns["root_id"].to_numpy(), name="drive")


def encode_i_ext(
    columns: pd.DataFrame,
    flow_vx: np.ndarray,
    flow_vy: np.ndarray,
    subnetwork_root_ids: np.ndarray,
    peak_weight: float = 30.0,
) -> np.ndarray:
    """Map a flow field to a per-neuron drive-weight vector aligned with a
    loaded subnetwork's neuron ordering (as returned by
    `flyguard.extract.load_subnetwork`).

    T4/T5 neurons not present in `columns` (e.g. because `column_assignment`
    doesn't cover every reconstructed neuron) or not present in the
    subnetwork get zero drive. `peak_weight` scales drive=1 to a synapse
    -equivalent weight; CLAUDE.md's "Known gotcha" puts the useful range at
    20-40 per neuron -- but that number is calibrated for a *Poisson
    impulse* (`net.poisson`/`force_spike`-style), not a value re-injected as
    `i_ext` on every single step. Feeding this vector in unchanged at every
    `net.step()` bypasses the exponential synaptic decay entirely and
    saturates most driven neurons regardless of the fine-grained drive
    differences between conditions (see CLAUDE.md pilot finding on the
    encoder). Use `poisson_stim_fn` below to drive a real LIFNetwork
    correctly; this function is the lower-level per-condition primitive it
    is built on.
    """
    drive = encode_drive(columns, flow_vx, flow_vy)
    i_ext = np.zeros(len(subnetwork_root_ids), dtype=np.float32)
    root_to_idx = {int(rid): i for i, rid in enumerate(subnetwork_root_ids)}
    for root_id, d in drive.items():
        idx = root_to_idx.get(int(root_id))
        if idx is not None:
            i_ext[idx] = d * peak_weight
    return i_ext


def poisson_stim_fn(
    columns: pd.DataFrame,
    flow_vx: np.ndarray,
    flow_vy: np.ndarray,
    subnetwork_root_ids: np.ndarray,
    dt: float,
    base_rate_hz: float = 250.0,
    peak_weight: float = 30.0,
    seed: int = 0,
):
    """Build a `stim_fn(step) -> i_ext` closure for `LIFNetwork.run`, the
    same Poisson-impulse methodology already proven in
    `flyguard.stimuli.simulate_condition` and `inspect_subnetwork.py`'s
    `net.poisson(...)`: each T4/T5 neuron receives a discrete external
    impulse of size `peak_weight` (in the 20-40 "Known gotcha" range) with
    instantaneous probability `drive * base_rate_hz * dt`, rather than a
    constant current re-injected every step. This keeps individual T4/T5
    input events subject to the network's own exponential synaptic decay,
    the same as any real synapse.
    """
    drive = encode_drive(columns, flow_vx, flow_vy)
    root_to_idx = {int(rid): i for i, rid in enumerate(subnetwork_root_ids)}
    idx_list, drive_list = [], []
    for root_id, d in drive.items():
        i = root_to_idx.get(int(root_id))
        if i is not None and d > 0:
            idx_list.append(i)
            drive_list.append(d)
    idx = np.array(idx_list, dtype=np.int64)
    prob = np.clip(np.array(drive_list, dtype=np.float64) * base_rate_hz * dt, 0, 1)
    n = len(subnetwork_root_ids)
    rng = np.random.default_rng(seed)

    def stim_fn(_step):
        i_ext = np.zeros(n, dtype=np.float32)
        if len(idx) == 0:
            return i_ext
        hits = rng.random(len(idx)) < prob
        i_ext[idx[hits]] = peak_weight
        return i_ext

    return stim_fn


# ---------------------------------------------------------------------------
# Real-image encoder: rendered pixels -> per-column flow (CLAUDE.md's
# repeatedly-deferred "real optic-flow-from-image-pair" step). Everything
# above this point works on synthetic flow fields; everything below maps
# `flyguard.optical_flow`'s pixel-space flow onto real T4/T5 columns so it
# can be fed into `encode_drive`/`poisson_stim_fn` exactly like the
# synthetic generators are.
#
# **Stated assumption, not yet validated (same status as the direction
# convention above -- flagged, not resolved):** the rendered camera's field
# of view is assumed to span exactly the (x, y) extent of the T4/T5 columns
# present in `columns` (e.g., one hemisphere's population), stretched
# linearly onto the image. There is no independent way to calibrate a real
# camera's FOV against this dataset's retinotopic coordinate system, so
# rather than invent an unvalidated "FOV fraction" parameter on top of the
# direction-convention assumption already open, this uses the simplest
# parameter-free choice: the full observed column range maps to the full
# image. A real fly's compound eye is panoramic; a ~45-degree pinhole
# camera is not, so in truth only a fraction of columns should ever see
# the rendered scene -- this mapping cannot represent that, and columns
# get flow-driven input across the whole image regardless of whether that
# location is anatomically plausible for a real camera's coverage.
# `validate_real_encoder.py` reports results under this explicit
# assumption; do not read its accuracy numbers as validating the mapping
# itself.
# ---------------------------------------------------------------------------


def column_pixel_bounds(columns: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    """(x_bounds, y_bounds) spanning the given columns -- the window the
    rendered image is assumed to cover (see module note above)."""
    x = columns["x"].to_numpy()
    y = columns["y"].to_numpy()
    return (float(x.min()), float(x.max())), (float(y.min()), float(y.max()))


def column_to_pixel_rect(x: np.ndarray, y: np.ndarray, height: int, width: int,
                          x_bounds: tuple[float, float], y_bounds: tuple[float, float],
                          flip_x: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Map column (x, y) coordinates linearly onto (col, row) pixel
    coordinates in an arbitrary `height x width` image region. Column y
    follows the "+y = dorsal/up" convention already fixed in this module
    (see `DIRECTION_UNIT_VECTOR`); image rows increase downward, so larger
    column y maps to a *smaller* row index.

    `flip_x` mirrors the horizontal mapping, so column +x (rostral/front,
    per this module's stated convention) lands at the *left* edge of the
    region rather than the right. That is needed for a hemifield whose
    rostral direction points toward the image centre rather than away from
    it -- see `flyguard.arena` for where this actually bites. It is off by
    default so every existing result in this repo keeps its exact previous
    behaviour.
    """
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    fx = (x - xmin) / max(xmax - xmin, 1e-9)
    if flip_x:
        fx = 1.0 - fx
    px = fx * (width - 1)
    py = (ymax - y) / max(ymax - ymin, 1e-9) * (height - 1)
    return px, py


def column_to_pixel(x: np.ndarray, y: np.ndarray, resolution: int,
                     x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Square-image special case of `column_to_pixel_rect`."""
    return column_to_pixel_rect(x, y, resolution, resolution, x_bounds, y_bounds)


def sample_flow_rect(vx_img: np.ndarray, vy_img: np.ndarray, columns: pd.DataFrame,
                      x_bounds: tuple[float, float], y_bounds: tuple[float, float],
                      flip_x: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """`sample_flow_at_columns` for a non-square image region (e.g. one
    half of a frame), with the optional horizontal mirror."""
    from scipy.ndimage import map_coordinates

    height, width = vx_img.shape[:2]
    px, py = column_to_pixel_rect(columns["x"].to_numpy(), columns["y"].to_numpy(),
                                   height, width, x_bounds, y_bounds, flip_x=flip_x)
    coords = np.array([py, px])
    flow_vx = map_coordinates(vx_img, coords, order=1, mode="nearest")
    flow_vy = map_coordinates(vy_img, coords, order=1, mode="nearest")
    return flow_vx, flow_vy


def sample_flow_at_columns(vx_img: np.ndarray, vy_img: np.ndarray, columns: pd.DataFrame,
                            x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample a pixel-space flow field (e.g. from
    `optical_flow.lucas_kanade_flow_iterative`) at each column's mapped
    image position. Returns (flow_vx, flow_vy), each aligned with
    `columns` rows -- the same shape/order `expanding_flow_2d` and
    `translational_flow_2d` already produce, so this is a drop-in
    replacement for those in `encode_drive`/`poisson_stim_fn`.
    """
    return sample_flow_rect(vx_img, vy_img, columns, x_bounds, y_bounds)


def flow_from_frame_pair(frame1: np.ndarray, frame2: np.ndarray, columns: pd.DataFrame,
                          x_bounds: tuple[float, float] | None = None,
                          y_bounds: tuple[float, float] | None = None,
                          window: int = 15, n_iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """End to end: two consecutive rendered RGB frames -> per-column flow,
    ready for `encode_drive`/`poisson_stim_fn`. Uses
    `optical_flow.lucas_kanade_flow_iterative` (real pixel motion, not a
    synthetic ground-truth flow field) and `sample_flow_at_columns`
    (the stated-assumption mapping above). `x_bounds`/`y_bounds` default to
    `column_pixel_bounds(columns)` if not given.
    """
    from flyguard.optical_flow import lucas_kanade_flow_iterative, to_grayscale

    if x_bounds is None or y_bounds is None:
        x_bounds, y_bounds = column_pixel_bounds(columns)
    g1, g2 = to_grayscale(frame1), to_grayscale(frame2)
    vx_img, vy_img = lucas_kanade_flow_iterative(g1, g2, window=window, n_iters=n_iters)
    return sample_flow_at_columns(vx_img, vy_img, columns, x_bounds, y_bounds)
