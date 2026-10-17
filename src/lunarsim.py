"""
LunarSim: a lightweight procedural renderer for lunar-surface place recognition.

Terrain:   regional DEM (16 m/px, 16 km) with fBm relief, a crater population drawn from
           an equilibrium size-frequency distribution N(>D) = c D^-2, and distant massifs;
           a local DEM (0.3 m/px, ~614 m) with fBm, small craters, ejecta boulders and
           crater-halo albedo, blended into the regional DEM in the centre of the map.
Photometry: lunar-Lambert (McEwen 1991) with phase-dependent limb-darkening L(g), a
           Hapke-style shadow-hiding opposition surge, cast shadows from both DEMs with a
           0.53 deg solar disk (penumbra), weak ambient (earthshine / multiple scattering),
           and a micro-relief bump texture faded by pixel footprint.
Camera:    pinhole, rover-mounted, zero roll/pitch with an off-centre principal point,
           lunar curvature, auto-exposure, gamma, shot/read noise, 8-bit quantisation.
Rendering is column-wise (heightfield "voxel-space" ray casting) and fully vectorised.
"""
import numpy as np
from numpy.fft import ifft2, fftfreq
from scipy import ndimage

R_MOON = 1737.4e3


# ----------------------------------------------------------------------------- terrain
def fbm(n, beta, rng):
    kx = fftfreq(n)[:, None]
    ky = fftfreq(n)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1.0
    amp = k ** (-beta / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, (n, n))
    h = np.real(ifft2(amp * np.exp(1j * phase)))
    return (h - h.mean()) / h.std()


def add_craters(H, A, res, rng, Dmin, Dmax, c=0.08, boulders=None):
    """Stamp craters with cumulative SFD N(>D)=c*D^-2 per m^2 (D in m).
    H: heights (m), A: albedo map (modified in place for fresh-crater halos)."""
    n = H.shape[0]
    area = (n * res) ** 2
    N = int(c * area * (Dmin ** -2 - Dmax ** -2))
    u = rng.random(N)
    D = (Dmin ** -2 - u * (Dmin ** -2 - Dmax ** -2)) ** -0.5
    D = np.sort(D)[::-1]                      # large (old) first, small superposed later
    fresh_all = rng.beta(1.2, 2.5, N)          # 1 = fresh, 0 = degraded
    for D_, fr in zip(D, fresh_all):
        R = D_ / 2
        cx, cy = rng.uniform(0, n * res, 2)
        w = int(np.ceil(3 * R / res)) + 1
        ic, jc = int(cy / res), int(cx / res)
        i0, i1, j0, j1 = max(ic - w, 0), min(ic + w + 1, n), max(jc - w, 0), min(jc + w + 1, n)
        if i0 >= i1 or j0 >= j1:
            continue
        yy = (np.arange(i0, i1) + 0.5) * res - cy
        xx = (np.arange(j0, j1) + 0.5) * res - cx
        r = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2) / R
        depth = D_ * (0.04 + 0.16 * fr)
        rim = D_ * (0.005 + 0.035 * fr)
        inside = r < 1
        prof = np.where(inside, rim - (depth + rim) * np.clip(1 - r ** 2, 0, None) ** (1.0 + 1.5 * (1 - fr)),
                        rim * np.clip(r, 1, None) ** -3)
        prof *= np.clip((3 - r) / 0.5, 0, 1)   # taper at window edge
        H[i0:i1, j0:j1] += prof
        if A is not None and fr > 0.55:
            halo = 0.25 * (fr - 0.55) / 0.45 * np.exp(-((r - 1.0) / 0.6) ** 2) * (r < 2.5)
            A[i0:i1, j0:j1] *= 1 + halo
        if boulders is not None and fr > 0.6 and D_ > 6:
            nb = int(rng.poisson(D_ * 0.6 * fr))
            for _ in range(nb):
                rr = R * (1 + rng.exponential(0.4))
                th = rng.uniform(0, 2 * np.pi)
                boulders.append((cx + rr * np.cos(th), cy + rr * np.sin(th),
                                 min(0.15 + rng.pareto(2.5) * 0.4, 3.0)))
    return H


def add_boulders(H, A, res, blist):
    n = H.shape[0]
    for bx, by, s in blist:
        sig = max(s / 2.0, res * 0.7)
        w = int(np.ceil(3 * sig / res)) + 1
        ic, jc = int(by / res), int(bx / res)
        i0, i1, j0, j1 = max(ic - w, 0), min(ic + w + 1, n), max(jc - w, 0), min(jc + w + 1, n)
        if i0 >= i1 or j0 >= j1:
            continue
        yy = (np.arange(i0, i1) + 0.5) * res - by
        xx = (np.arange(j0, j1) + 0.5) * res - bx
        g = np.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) / (2 * sig ** 2))
        H[i0:i1, j0:j1] += 0.7 * s * g
        A[i0:i1, j0:j1] *= 1 + 0.25 * g


class Terrain:
    def __init__(self, seed, n_near=2048, res_near=0.3, n_reg=1024, res_reg=16.0):
        rng = np.random.default_rng(seed)
        self.seed = seed
        # regional ---------------------------------------------------------
        self.res_reg, self.n_reg = res_reg, n_reg
        L = n_reg * res_reg
        Hr = 30.0 * fbm(n_reg, 3.2, rng)
        Ar = np.ones((n_reg, n_reg), np.float64) * 0.12
        Ar *= 1 + 0.12 * fbm(n_reg, 2.5, rng)
        add_craters(Hr, Ar, res_reg, rng, Dmin=60, Dmax=2500, c=0.05)
        # massifs at 2.5-7 km from the centre
        yy, xx = np.meshgrid((np.arange(n_reg) + 0.5) * res_reg - L / 2,
                             (np.arange(n_reg) + 0.5) * res_reg - L / 2, indexing="ij")
        ridged = np.abs(fbm(n_reg, 2.6, rng))
        for _ in range(rng.integers(4, 8)):
            rad = rng.uniform(4000, 7500)
            th = rng.uniform(0, 2 * np.pi)
            mx, my = rad * np.cos(th), rad * np.sin(th)
            sx, sy = rng.uniform(900, 2500, 2)
            ang = rng.uniform(0, np.pi)
            xr = (xx - mx) * np.cos(ang) + (yy - my) * np.sin(ang)
            yr = -(xx - mx) * np.sin(ang) + (yy - my) * np.cos(ang)
            Hr += rng.uniform(200, 700) * np.exp(-(xr / sx) ** 2 - (yr / sy) ** 2) * (0.9 + 0.1 * ridged)
        # flatten the central valley a little (landing-site like)
        rc = np.sqrt(xx ** 2 + yy ** 2)
        Hr *= 0.12 + 0.88 * np.clip((rc - 600) / 2000, 0, 1)
        self.Hr = Hr
        self.Ar = Ar
        # local --------------------------------------------------------------
        self.res_n, self.n_n = res_near, n_near
        Ln = n_near * res_near
        self.Ln = Ln
        Hn = 0.6 * fbm(n_near, 3.2, rng) + 0.06 * fbm(n_near, 2.2, rng)
        An = np.ones((n_near, n_near)) * (1 + 0.08 * fbm(n_near, 2.4, rng))
        bl = []
        add_craters(Hn, An, res_near, rng, Dmin=1.5, Dmax=120, c=0.06, boulders=bl)
        for _ in range(int(Ln * Ln * 0.004)):  # scattered background boulders
            bl.append((rng.uniform(0, Ln), rng.uniform(0, Ln), min(0.15 + rng.pareto(3.0) * 0.3, 1.5)))
        add_boulders(Hn, An, res_near, bl)
        # blend window (1 in the central 80 %, 0 at edges)
        t = (np.arange(n_near) + 0.5) / n_near
        w1 = np.clip(np.minimum(t, 1 - t) / 0.1, 0, 1)
        w1 = 0.5 - 0.5 * np.cos(np.pi * w1)
        W = w1[:, None] * w1[None, :]
        # near grid coordinates in world frame (centred at origin)
        self.near_origin = -Ln / 2
        cn = (np.arange(n_near) + 0.5) * res_near - Ln / 2
        Hreg_on_near = self._sample(self.Hr, cn[None, :].repeat(n_near, 0), cn[:, None].repeat(n_near, 1),
                                    -L / 2, res_reg)
        Areg_on_near = self._sample(self.Ar, cn[None, :].repeat(n_near, 0), cn[:, None].repeat(n_near, 1),
                                    -L / 2, res_reg)
        self.Hn = Hreg_on_near + W * Hn
        self.An = Areg_on_near * (1 + W * (An - 1))
        self.reg_origin = -L / 2
        # normals (gradients) ---------------------------------------------------
        self.gyr, self.gxr = np.gradient(self.Hr, res_reg)
        self.gyn, self.gxn = np.gradient(self.Hn, res_near)
        # micro-relief bump textures (two incommensurate tiles)
        self.micro = [(fbm(1024, 2.4, rng), 0.025), (fbm(1024, 2.4, rng), 0.0331)]
        self.micro_mid = [(fbm(512, 2.2, rng), 0.21)]
        self._sun_cache = {}

    @staticmethod
    def _sample(M, x, y, origin, res):
        j = (x - origin) / res - 0.5
        i = (y - origin) / res - 0.5
        return ndimage.map_coordinates(M, [i, j], order=1, mode="nearest")

    def height(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        hr = self._sample(self.Hr, x, y, self.reg_origin, self.res_reg)
        inn = (np.abs(x) < self.Ln / 2 - 1) & (np.abs(y) < self.Ln / 2 - 1)
        out = hr.copy()
        if inn.any():
            out[inn] = self._sample(self.Hn, x[inn], y[inn], self.near_origin, self.res_n)
        return out

    # ------------------------------------------------------------------- shadows
    @staticmethod
    def _shadow(H, res, az, el):
        """Binary cast-shadow mask of heightfield H for sun azimuth az (rad, CCW from +x) and
        elevation el (rad). Column index j ~ +x, row index i ~ +y."""
        ang = np.degrees(az)
        # rotate so that the sun direction maps to +columns
        Hrot = ndimage.rotate(H, ang, reshape=True, order=1, mode="constant", cval=-1e9)
        n = Hrot.shape[1]
        x = np.arange(n) * res
        g = Hrot - x[None, :] * np.tan(el)
        valid = Hrot > -1e8
        g = np.where(valid, g, -np.inf)
        # running max over cells closer to the sun (to the right), exclusive
        rev = np.maximum.accumulate(g[:, ::-1], axis=1)[:, ::-1]
        ahead = np.concatenate([rev[:, 1:], np.full((g.shape[0], 1), -np.inf)], axis=1)
        sh = (g < ahead - 1e-6).astype(np.float32)
        back = ndimage.rotate(sh, -ang, reshape=True, order=1, mode="constant", cval=0)
        # crop centre to the original size
        oi = (back.shape[0] - H.shape[0]) // 2
        oj = (back.shape[1] - H.shape[1]) // 2
        return back[oi:oi + H.shape[0], oj:oj + H.shape[1]]

    def sun_maps(self, az_deg, el_deg):
        key = (round(az_deg, 3), round(el_deg, 3))
        if key in self._sun_cache:
            return self._sun_cache[key]
        az, el = np.radians(az_deg), np.radians(el_deg)
        half = np.radians(0.265)  # solar disk radius
        els = [max(el - half, np.radians(0.05)), el, el + half]
        # (sign verified by unit test: shadow lies on the anti-sun side for all azimuths)
        shr = np.mean([self._shadow(self.Hr, self.res_reg, az, e) for e in els], axis=0)
        shn = np.mean([self._shadow(self.Hn, self.res_n, az, e) for e in els], axis=0)
        # regional shadows (e.g. from massifs) also darken the local patch
        Ln, n = self.Ln, self.n_n
        cn = (np.arange(n) + 0.5) * self.res_n - Ln / 2
        shr_on_n = self._sample(shr, cn[None, :].repeat(n, 0), cn[:, None].repeat(n, 1),
                                self.reg_origin, self.res_reg)
        shn = np.maximum(shn, shr_on_n)
        out = (shr.astype(np.float32), shn.astype(np.float32))
        self._sun_cache = {key: out}  # keep only the latest (memory)
        return out


# ----------------------------------------------------------------------------- render
def lunar_lambert_L(g_deg):
    # McEwen (1991) phase-dependent limb-darkening parameter
    L = 1.0 - 0.019 * g_deg + 2.42e-4 * g_deg ** 2 - 1.46e-6 * g_deg ** 3
    return np.clip(L, 0.0, 1.0)


def render(T, x0, y0, yaw_deg, az_deg, el_deg, W=320, Hh=240, hfov_deg=70.0, cam_h=1.5,
           cy_frac=0.30, rng=None, noise=True, return_depth=False):
    rng = np.random.default_rng() if rng is None else rng
    f = (W / 2) / np.tan(np.radians(hfov_deg) / 2)
    cx, cy = W / 2 - 0.5, Hh * cy_frac
    u = np.arange(W)
    alpha = np.arctan((u - cx) / f)                     # column azimuth offset
    theta = np.radians(yaw_deg) - alpha                 # world azimuth of each column (u right = cw)
    zcam = float(T.height(np.array([x0]), np.array([y0]))[0]) + cam_h
    # distance samples (geometric growth)
    d = [0.8]
    while d[-1] < 12000:
        d.append(d[-1] * 1.0045 + 0.012)
    d = np.array(d)
    S = len(d)
    X = x0 + d[:, None] * np.cos(theta)[None, :]
    Y = y0 + d[:, None] * np.sin(theta)[None, :]
    Hs = T.height(X.ravel(), Y.ravel()).reshape(S, W) - d[:, None] ** 2 / (2 * R_MOON)
    zc = d[:, None] * np.cos(alpha)[None, :]
    v = cy - f * (Hs - zcam) / zc                       # image row of each sample
    vmin = np.minimum.accumulate(v, axis=0)
    rows = np.arange(Hh) + 0.5
    hit_s = np.empty((Hh, W), int)
    for c in range(W):
        hit_s[:, c] = np.searchsorted(-vmin[:, c], -rows, side="left")
    sky = (hit_s >= S) | (hit_s == 0)
    s1 = np.clip(hit_s, 1, S - 1)
    cc = np.broadcast_to(np.arange(W)[None, :], (Hh, W))
    v1, v0 = v[s1, cc], v[s1 - 1, cc]
    t = np.clip((v0 - rows[:, None]) / np.maximum(v0 - v1, 1e-9), 0, 1)
    dh = d[s1 - 1] + t * (d[s1] - d[s1 - 1])
    th = theta[None, :].repeat(Hh, 0)
    Px = x0 + dh * np.cos(th)
    Py = y0 + dh * np.sin(th)
    Pz = Hs[s1 - 1, cc] + t * (Hs[s1, cc] - Hs[s1 - 1, cc])

    # surface attributes
    shr, shn = T.sun_maps(az_deg, el_deg)
    inn = (np.abs(Px) < T.Ln / 2 - 1) & (np.abs(Py) < T.Ln / 2 - 1)
    gx = T._sample(T.gxr, Px, Py, T.reg_origin, T.res_reg)
    gy = T._sample(T.gyr, Px, Py, T.reg_origin, T.res_reg)
    alb = T._sample(T.Ar, Px, Py, T.reg_origin, T.res_reg)
    shad = T._sample(shr, Px, Py, T.reg_origin, T.res_reg)
    if inn.any():
        gx[inn] = T._sample(T.gxn, Px[inn], Py[inn], T.near_origin, T.res_n)
        gy[inn] = T._sample(T.gyn, Px[inn], Py[inn], T.near_origin, T.res_n)
        alb[inn] = T._sample(T.An, Px[inn], Py[inn], T.near_origin, T.res_n) * 0.12
        shad[inn] = T._sample(shn, Px[inn], Py[inn], T.near_origin, T.res_n)
    # pixel footprint (m) and micro-relief bump mapping
    foot = dh / f / np.maximum(np.cos(np.arctan((rows[:, None] - cy) / f)), 0.2)
    for tex, cell, amp, fade in [(T.micro[0][0], T.micro[0][1], 0.010, 0.06),
                                 (T.micro[1][0], T.micro[1][1], 0.010, 0.08),
                                 (T.micro_mid[0][0], T.micro_mid[0][1], 0.05, 0.5)]:
        n_t = tex.shape[0]
        jj = (Px / cell) % n_t
        ii = (Py / cell) % n_t
        ty, tx = np.gradient(tex)
        fw = np.clip(1 - foot / fade, 0, 1)
        k = fw * amp / cell
        gx = gx + k * ndimage.map_coordinates(tx, [ii, jj], order=1, mode="wrap")
        gy = gy + k * ndimage.map_coordinates(ty, [ii, jj], order=1, mode="wrap")
        alb = alb * (1 + 0.08 * fw * ndimage.map_coordinates(tex, [ii, jj], order=1, mode="wrap"))
    nrm = np.stack([-gx, -gy, np.ones_like(gx)], 0)
    nrm /= np.linalg.norm(nrm, axis=0, keepdims=True)
    az, el = np.radians(az_deg), np.radians(el_deg)
    Ls = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    V = np.stack([x0 - Px, y0 - Py, zcam - Pz], 0)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    cos_i = np.clip(np.tensordot(Ls, nrm, axes=1), 0, None) * (1 - shad)
    cos_e = np.clip((nrm * V).sum(0), 0.02, None)
    g = np.degrees(np.arccos(np.clip(np.tensordot(Ls, V, axes=1), -1, 1)))
    Lg = lunar_lambert_L(g)
    refl = 2 * Lg * cos_i / (cos_i + cos_e + 1e-6) + (1 - Lg) * cos_i
    B = 1.0 / (1 + np.tan(np.radians(g) / 2) / 0.06)   # opposition surge (B0 = 1, h = 0.06)
    I = alb * refl * (1 + B) + alb * 0.012 * (0.5 + 0.5 * nrm[2])   # weak ambient
    I = np.where(sky, 0.0, I)
    # camera: auto exposure, vignetting, gamma, noise, quantisation
    # auto-exposure metered on the lower (ground) part of the frame, as a rover camera would
    meter = I[int(Hh * 0.45):][~sky[int(Hh * 0.45):]]
    ref = np.median(meter) if meter.size > 100 else 1.0
    img = I / max(ref, 1e-6) * 0.20     # centre-weighted "middle grey" metering
    yy, xx = np.mgrid[0:Hh, 0:W]
    rr = ((xx - W / 2) ** 2 + (yy - Hh / 2) ** 2) / (W / 2) ** 2
    img = img * (1 - 0.12 * rr)
    img = (1 - np.exp(-1.6 * np.clip(img, 0, None))) ** (1 / 2.2)   # soft-shoulder tone curve + gamma
    if noise:
        img = img + rng.normal(0, 0.012, img.shape) + rng.normal(0, 1, img.shape) * 0.01 * np.sqrt(img)
    img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    if return_depth:
        return img, np.where(sky, np.inf, dh)
    return img
