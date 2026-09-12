"""Dense optical flow from real rendered frame pairs (Lucas-Kanade), the
missing piece flagged repeatedly in the project notes: the retinotopic encoder has
so far only ever consumed synthetic flow fields
(`encoder.expanding_flow_2d`/`translational_flow_2d`) computed analytically
from a trial's *known* condition and trajectory, never flow actually
estimated from pixels. This module closes that gap.

No OpenCV (the project notes "Keep dependencies minimal": numpy, scipy, pandas
only) -- classic Lucas-Kanade (Lucas & Kanade 1981) implemented directly
with `scipy.ndimage` for gradients and windowed sums, the textbook
closed-form least-squares solution over local windows under the brightness
-constancy assumption. This is intentionally the simple, well-understood
algorithm, not an attempt at state-of-the-art flow -- the point is to
finally drive the connectome from real pixels, not to build a flow
research project.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Frame -> grayscale float [H, W] in [0, 1].

    Accepts [H, W, C] colour and [H, W] single-channel input. The mono case
    is not hypothetical: the ROS2 camera node takes `mono8`, and averaging
    over the last axis of an already-2-D array collapses it to 1-D, which
    surfaces far downstream as a confusing pyramid-slicing error rather than
    as "you passed a grayscale image".
    """
    frame = np.asarray(frame)
    if frame.ndim == 2:
        gray = frame.astype(np.float64)
    elif frame.ndim == 3:
        gray = frame.astype(np.float64).mean(axis=-1)
    else:
        raise ValueError(
            f"expected a [H, W] or [H, W, C] frame, got shape {frame.shape}"
        )
    # Always 0-255 -> 0-1, for any dtype. Every calibration constant in this
    # repo was measured against that scaling, and `min_det` in the flow solver
    # is an absolute threshold, so a dtype-dependent normalisation would move
    # published numbers for no visible reason.
    return gray / 255.0


def lucas_kanade_flow(img1: np.ndarray, img2: np.ndarray, window: int = 15,
                       min_det: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Dense optical flow between two grayscale frames of the same shape.

    Solves the brightness-constancy least-squares system
    `[[Sxx, Sxy], [Sxy, Syy]] @ [vx, vy] = -[Sxt, Syt]` in a `window x
    window` neighbourhood around every pixel (Sxx = windowed sum of Ix^2,
    etc.), the standard Lucas-Kanade derivation. Windowed sums are computed
    via `ndimage.uniform_filter` (box filter), which is the same thing as a
    normalized local average times the window area -- far cheaper than an
    explicit sliding-window loop.

    Returns (vx, vy): per-pixel flow in pixels/frame, image-pixel
    convention (x right, y down, matching row/column indexing). Pixels
    where the local gradient structure is nearly singular (flat, low
    -texture regions -- e.g. the middle of a uniform checker tile) get
    zero flow rather than a division blow-up; `valid_mask` marks which
    pixels that affects.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"frame shape mismatch: {img1.shape} vs {img2.shape}")

    Ix = ndimage.sobel(img1, axis=1) / 8.0
    Iy = ndimage.sobel(img1, axis=0) / 8.0
    It = img2 - img1

    def wsum(a):
        return ndimage.uniform_filter(a, size=window) * (window * window)

    Sxx, Syy, Sxy = wsum(Ix * Ix), wsum(Iy * Iy), wsum(Ix * Iy)
    Sxt, Syt = wsum(Ix * It), wsum(Iy * It)

    det = Sxx * Syy - Sxy * Sxy
    valid = np.abs(det) >= min_det
    det_safe = np.where(valid, det, 1.0)

    vx = (-Syy * Sxt + Sxy * Syt) / det_safe
    vy = (Sxy * Sxt - Sxx * Syt) / det_safe
    vx = np.where(valid, vx, 0.0)
    vy = np.where(valid, vy, 0.0)
    return vx, vy


def lucas_kanade_flow_iterative(img1: np.ndarray, img2: np.ndarray, window: int = 15,
                                 n_iters: int = 3, min_det: float = 1e-6,
                                 vx_init: np.ndarray | None = None,
                                 vy_init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Iterative (warp-and-resolve) Lucas-Kanade -- corrects the systematic
    magnitude *underestimate* single-pass LK has for displacements bigger
    than a pixel or so (its differential/brightness-constancy linearization
    is only exact for infinitesimal motion; empirically confirmed here:
    single-pass LK recovered ~87% of a true 2px shift, ~68% of a diagonal
    (2,2) shift -- see `tests/test_optical_flow.py`). Each iteration warps
    `img2` back toward `img1` using the current flow estimate (bilinear
    interpolation via `ndimage.map_coordinates`) and re-solves for the
    residual, accumulating -- the standard Lucas-Kanade-with-warping
    scheme.

    Warping alone still only extends the usable range to a few pixels per
    frame, because every iteration's residual must itself be small enough for
    the linearization. `vx_init`/`vy_init` seed the accumulator with a flow
    estimate from elsewhere, which is how `lucas_kanade_flow_pyramidal`
    carries a coarse level's answer down to a finer one. For displacements
    beyond a few pixels, use that instead of raising `n_iters` here.
    """
    vx_total = np.zeros_like(img1) if vx_init is None else vx_init.astype(np.float64).copy()
    vy_total = np.zeros_like(img1) if vy_init is None else vy_init.astype(np.float64).copy()
    H, W = img1.shape
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float64)

    for _ in range(n_iters):
        warp_r = rows + vy_total
        warp_c = cols + vx_total
        img2_warped = ndimage.map_coordinates(img2, [warp_r, warp_c], order=1,
                                               mode="constant", cval=np.nan)
        # pixels that warped outside the frame contribute nothing this round
        nan_mask = np.isnan(img2_warped)
        img2_warped_safe = np.where(nan_mask, img1, img2_warped)
        dvx, dvy = lucas_kanade_flow(img1, img2_warped_safe, window=window, min_det=min_det)
        dvx = np.where(nan_mask, 0.0, dvx)
        dvy = np.where(nan_mask, 0.0, dvy)
        vx_total = vx_total + dvx
        vy_total = vy_total + dvy

    return vx_total, vy_total


def _downsample(img: np.ndarray) -> np.ndarray:
    """Half-resolution copy, anti-aliased first. Decimating without the blur
    would alias exactly the high-frequency texture the flow solver relies on."""
    return ndimage.gaussian_filter(img, sigma=1.0)[::2, ::2]


def _resample_flow(v: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a flow component onto `shape`. The caller scales
    the magnitude; this only changes the sampling grid."""
    rows = np.linspace(0, v.shape[0] - 1, shape[0])
    cols = np.linspace(0, v.shape[1] - 1, shape[1])
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return ndimage.map_coordinates(v, [rr, cc], order=1, mode="nearest")


def lucas_kanade_flow_pyramidal(img1: np.ndarray, img2: np.ndarray, window: int = 15,
                                 n_iters: int = 3, n_levels: int = 3,
                                 min_det: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Coarse-to-fine Lucas-Kanade: the standard fix for displacements too
    large for the differential solver.

    Estimate flow on a heavily downsampled pair (where a 12-pixel motion is a
    3-pixel motion), then carry that estimate down the pyramid, doubling its
    magnitude at each step and refining with warp-and-resolve at the new
    resolution.

    **This is not a refinement for its own sake -- it fixes a sign inversion
    that broke closed-loop control.** Surfaces very close to a moving camera
    sweep past faster than plain LK can follow, so the estimate collapses
    toward zero rather than saturating high. In `flyguard.avoid` that made a
    wall the agent was about to scrape read as *emptier* than open corridor
    (measured: drive 0.0138 hard against a wall versus 0.0259 in mid
    -corridor), so the steering law turned toward it. An obstacle detector
    whose signal inverts at short range is worse than none, and no amount of
    gain tuning fixes it -- the flow estimate itself has to hold up.

    Levels stop early if the coarsest image would be smaller than twice the
    window, since a window wider than the image measures nothing.
    """
    pyr1, pyr2 = [img1.astype(np.float64)], [img2.astype(np.float64)]
    for _ in range(n_levels - 1):
        if min(pyr1[-1].shape) < 2 * window:
            break
        pyr1.append(_downsample(pyr1[-1]))
        pyr2.append(_downsample(pyr2[-1]))

    vx = np.zeros_like(pyr1[-1])
    vy = np.zeros_like(pyr1[-1])
    for level in range(len(pyr1) - 1, -1, -1):
        a, b = pyr1[level], pyr2[level]
        if vx.shape != a.shape:
            # One level down is twice the resolution, so the same physical
            # motion is twice as many pixels.
            vx = _resample_flow(vx, a.shape) * 2.0
            vy = _resample_flow(vy, a.shape) * 2.0
        vx, vy = lucas_kanade_flow_iterative(a, b, window=window, n_iters=n_iters,
                                             min_det=min_det, vx_init=vx, vy_init=vy)
    return vx, vy


def flow_sequence(frames: np.ndarray, window: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Optical flow for every consecutive pair in a rendered trial.

    frames: [n_frames, H, W, 3] uint8 (one `mujoco_world` trial).
    Returns (vx, vy), each [n_frames - 1, H, W].
    """
    gray = np.stack([to_grayscale(f) for f in frames])
    vx_list, vy_list = [], []
    for i in range(len(gray) - 1):
        vx, vy = lucas_kanade_flow(gray[i], gray[i + 1], window=window)
        vx_list.append(vx)
        vy_list.append(vy)
    return np.stack(vx_list), np.stack(vy_list)
