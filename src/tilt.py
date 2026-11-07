"""Rolled and pitched camera views for LunarSim.

The LunarSim renderer casts rays column by column, which is exact for a pinhole camera with zero roll
and pitch. A camera that is additionally rotated about its centre (roll about the optical axis, pitch
about the horizontal image axis) sees the same rays, only re-projected: its image is related to an
upright image with the same focal length by the homography K R K^-1, independently of scene depth.
We therefore render an upright view that is wide enough to contain every ray of the tilted view (same
focal length, same camera position), resample it through the homography, and only then apply the
camera model (auto-exposure, vignetting, tone curve, noise) to the final 320x240 frame, exactly as for
an upright benchmark image. With roll = pitch = 0 this reproduces `lunarsim.render` exactly.

Conventions: camera frame x right, y down, z forward. Roll rotates the camera about its central viewing
direction (see AXIS); positive pitch tilts the camera up, towards the sky.
"""
import numpy as np
from scipy import ndimage
from lunarsim import render, camera

W, H, HFOV, CY_FRAC = 320, 240, 70.0, 0.30
F = (W / 2) / np.tan(np.radians(HFOV) / 2)
CX, CY = W / 2 - 0.5, H * CY_FRAC          # pixel conventions of lunarsim.render: column u <-> x = u,
NEAR = list(np.geomspace(0.08, 0.79, 48))   # row r <-> y = r + 0.5


def _axis_rot(n, ang):
    n = n / np.linalg.norm(n)
    Kx = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx


# Roll is a rotation about the camera's central viewing direction (the ray through the image centre),
# so the same part of the scene stays in view. Because the principal point lies above the image centre,
# this axis points about 12 deg below the optical axis; rolling about the optical axis itself would swing
# the view towards the sky and confound orientation with what is visible.
AXIS = np.array([0.0, (H / 2 - CY) / F, 1.0])


def rotation(roll_deg, pitch_deg):
    """R maps a ray in the tilted camera frame to the upright camera frame."""
    b = np.radians(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return Rx @ _axis_rot(AXIS, np.radians(roll_deg))


def _rays(R, uu, rr):
    c = np.stack([(uu - CX) / F, (rr + 0.5 - CY) / F, np.ones_like(uu, dtype=float)], 0)
    d = np.tensordot(R, c, axes=1)
    assert (d[2] > 0.05).all(), 'tilt too large for a single upright source view'
    return F * d[0] / d[2], F * d[1] / d[2], d          # upright-image coordinates relative to the axis


def render_tilted(T, x0, y0, yaw_deg, az_deg, el_deg, cam_h=1.5, roll=0.0, pitch=0.0, rng=None,
                  noise=True, derotate=False, ss=3):
    """Render the 320x240 view of a camera with the given roll and pitch.

    The upright source view is rendered with ss-fold supersampling (focal length ss*F) and point-sampled
    through the homography with bilinear interpolation on that fine grid, so the tilted image is as
    sharp as a native render (interpolation smoothing <= 1/ss px). The identity case (no tilt) is
    rendered at ss = 1 and is exactly the benchmark renderer.
    derotate=True simulates gravity-based correction of the roll: the upright view, masked to the part
    of the scene that the rolled camera actually sees (black elsewhere), as an IMU-levelled pipeline
    would produce after rotating the rolled frame back. It needs no resampling."""
    rng = np.random.default_rng() if rng is None else rng
    uu, rr = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    R = np.eye(3) if derotate else rotation(roll, pitch)
    s = 1 if np.allclose(R, np.eye(3)) else int(ss)
    border = np.zeros((H, W), bool); border[[0, -1], :] = True; border[:, [0, -1]] = True
    xs, ys, d = _rays(R, uu[border], rr[border])
    xs, ys = s * xs, s * ys                              # in source pixels (focal length s*F)
    m = 3 * s
    # source column j = cx_w + X', row i = cy_w + Y' - 0.5 (lunarsim.render conventions). These offsets
    # put the native sample positions exactly on source pixels, so the identity case involves no blur.
    kx = m - int(np.floor(xs.min() + s * CX)); ky = m - int(np.floor(ys.min() + s * CY - 0.5))
    cx_w = s * CX + kx
    cy_w = s * CY - s / 2 + 0.5 + ky
    Ww = int(np.ceil(xs.max() + s * CX)) + kx + m + 1
    Hw = int(np.ceil(ys.max() + cy_w - 0.5)) + m + 1
    hfov_w = 2 * np.degrees(np.arctan(Ww / (2 * s * F)))
    # steepest ray below the horizon -> nearest ground hit; add near samples only when needed
    dep = np.arctan2(d[1], np.hypot(d[0], d[2])).max()
    near = NEAR if (dep > 0 and cam_h / np.tan(dep) < 1.0) else None
    I_w, sky_w, _ = render(T, x0, y0, yaw_deg, az_deg, el_deg, W=Ww, Hh=Hw, hfov_deg=hfov_w, cam_h=cam_h,
                           cy_frac=cy_w / Hw, rng=rng, noise=False, near=near, raw=True, cx=cx_w, lod=s)
    assert abs((Ww / 2) / np.tan(np.radians(hfov_w) / 2) - s * F) < 1e-6
    X, Y, _ = _rays(R, uu, rr)
    cols, rows = cx_w + s * X, cy_w + s * Y - 0.5
    I = ndimage.map_coordinates(I_w, [rows, cols], order=1, mode='constant', cval=0.0)
    sky = ndimage.map_coordinates(sky_w.astype(float), [rows, cols], order=1, mode='constant', cval=1.0) > 0.5
    if derotate and roll != 0:
        Rr = rotation(roll, 0)                            # rolled-camera ray -> upright ray
        c = np.stack([(uu - CX) / F, (rr + 0.5 - CY) / F, np.ones_like(uu)], 0)
        dr = np.tensordot(Rr.T, c, axes=1)               # upright pixel ray in the rolled camera's frame
        with np.errstate(divide='ignore', invalid='ignore'):
            cu, cr = CX + F * dr[0] / dr[2], CY + F * dr[1] / dr[2] - 0.5
        inside = (dr[2] > 0) & (cu >= -0.5) & (cu <= W - 0.5) & (cr >= -0.5) & (cr <= H - 0.5)
        I = np.where(inside, I, 0.0)
        sky = sky | ~inside
    return camera(I, sky, rng, noise)
