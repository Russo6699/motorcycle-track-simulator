"""Road geometry from satellite imagery — the method of the original project (computer vision + B-spline).

Input is a satellite image "unrolled" along the marked route: row r is a station every `row_step` metres along
the route, column q is a lateral offset of (q - (cols-1)/2) * col_res metres (positive = left of travel).
The map route is used only to know roughly where to look; the road itself is found in the pixels.

1. OpenCV: colour space conversion (Lab), smoothing, Sobel gradients.
2. Road colour model learned from the image itself (median colour along the route, running median).
3. For every station, every candidate centre c and half-width h is scored: asphalt-likeness inside the band,
   contrast against the shoulders, and edge strength at both borders.
4. Viterbi dynamic programming picks the most consistent centre and width along the whole route
   (continuity in both), so short gaps under trees or shadows are bridged. A second pass re-centres the image
   on the road found and searches again with the colour learned locally. Where one edge of the road is not
   visible (junctions, lay-bys) the centre is bridged from the reliable stretches on both sides.
5. scipy.interpolate.splprep fits a smoothing B-spline through the detected centre points, which is then
   sampled at `out_n` (default 50,000) equidistant points; curvature is evaluated with
   k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2).
"""
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d, median_filter


def _road_likelihood(lab, centre_col, col_res, row_step, window_m):
    """per-pixel similarity to the road colour, learned from a ±1 m strip around `centre_col` (per row).
    window_m=None → one colour model for the whole route (robust when the map line is off the road locally);
    otherwise a running median over ±window_m/2 along the road."""
    rows, cols, _ = lab.shape
    band = max(1, int(round(1.0 / col_res)))
    cc = np.clip(np.round(centre_col).astype(int), band, cols - band - 1)
    idx = cc[:, None] + np.arange(-band, band + 1)[None, :]
    strip = np.take_along_axis(lab, idx[:, :, None], axis=1)
    if window_m is None:
        model = np.broadcast_to(np.median(strip.reshape(-1, 3), axis=0), (rows, 3)).copy()
    else:
        model = np.median(strip, axis=1)
        win = max(3, int(round(window_m / row_step)) | 1)
        for ch in range(3):
            model[:, ch] = median_filter(model[:, ch], size=win, mode='nearest')
    dev = strip - model[:, None, :]
    mad = np.median(np.abs(dev).reshape(-1, 3), axis=0) * 1.4826
    sig = np.maximum(mad, [8.0, 3.0, 3.0])                         # floors in OpenCV Lab units
    d2 = (((lab - model[:, None, :]) / sig) ** 2).sum(axis=2)
    p = np.exp(-0.5 * np.minimum(d2, 50.0)).astype(np.float32)
    return gaussian_filter1d(p, sigma=max(1.0, 2.0 / row_step), axis=0), model, sig


def _scores(p, edge, r0, r1, cidx, hcols, mcols, w_edge):
    """score[r, ci, hi] for rows r0..r1 (vectorised with cumulative sums along columns)"""
    P = p[r0:r1]
    E = edge[r0:r1]
    cs = np.concatenate([np.zeros((P.shape[0], 1), np.float32), np.cumsum(P, axis=1)], axis=1)
    rows = P.shape[0]
    out = np.empty((rows, len(cidx), len(hcols)), np.float32)
    ncol = P.shape[1]
    for hi, h in enumerate(hcols):
        a = cidx - h
        b = cidx + h
        inside = (cs[:, b + 1] - cs[:, a]) / (2 * h + 1)
        la = np.clip(a - mcols, 0, ncol)
        rb = np.clip(b + 1 + mcols, 0, ncol)
        left = (cs[:, a] - cs[:, la]) / np.maximum(a - la, 1)
        right = (cs[:, rb] - cs[:, b + 1]) / np.maximum(rb - b - 1, 1)
        outside = 0.5 * (left + right)
        eg = 0.5 * (E[:, a] + E[:, b])
        out[:, :, hi] = 0.6 * inside + (inside - outside) + w_edge * eg
    return out


def _edges(lab, p, kr, row_step):
    L = cv2.GaussianBlur(lab[:, :, 0], (1, kr), sigmaX=0, sigmaY=max(0.5, 1.5 / row_step))
    gL = np.abs(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3))
    gP = np.abs(cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3))
    edge = gL / (np.percentile(gL, 95) + 1e-6) + gP / (np.percentile(gP, 95) + 1e-6)
    edge = cv2.dilate(edge, np.ones((1, 3), np.uint8))           # tolerate ±1 column
    return np.minimum(edge, 2.0) * 0.5


def _detect(p, edge, cols, col_res, row_step, max_offset, min_half, max_half, a_c, a_h, w_prior):
    """Viterbi over (centre, half-width); returns centre offset (m, sub-column), half-width index, confidence"""
    rows = p.shape[0]
    centre = (cols - 1) // 2
    cmax = int(round(max_offset / col_res))
    cidx = np.arange(centre - cmax, centre + cmax + 1)
    hcols = np.arange(int(round(min_half / col_res)), int(round(max_half / col_res)) + 1)
    mcols = max(2, int(round(1.5 / col_res)))
    cidx = cidx[(cidx - hcols[-1] - mcols >= 0) & (cidx + hcols[-1] + mcols < cols)]
    C, H = len(cidx), len(hcols)
    off_m = (cidx - (cols - 1) / 2.0) * col_res
    prior = (w_prior * off_m ** 2).astype(np.float32)[:, None]
    back = np.zeros((rows, C, H), np.int8)
    best_score = np.zeros(rows, np.float32)
    second = np.zeros(rows, np.float32)
    acc = None
    CH = 512
    for r0 in range(0, rows, CH):
        r1 = min(rows, r0 + CH)
        S = _scores(p, edge, r0, r1, cidx, hcols, mcols, 0.8) - prior
        for k in range(r1 - r0):
            s = S[k]
            if acc is None:
                acc = s.copy()
                back[r0 + k] = 4
            else:
                cand = np.full((9, C, H), -1e9, np.float32)
                j = 0
                for dc in (-1, 0, 1):
                    for dh in (-1, 0, 1):
                        cost = a_c * abs(dc) + a_h * abs(dh)
                        cs0, cs1 = max(0, dc), C + min(0, dc)
                        hs0, hs1 = max(0, dh), H + min(0, dh)
                        cand[j, cs0:cs1, hs0:hs1] = acc[cs0 - dc:cs1 - dc, hs0 - dh:hs1 - dh] - cost
                        j += 1
                arg = np.argmax(cand, axis=0)
                acc = np.take_along_axis(cand, arg[None], 0)[0] + s
                back[r0 + k] = arg.astype(np.int8)
            rowbest = s.max(axis=1)
            best_score[r0 + k] = rowbest.max()
            second[r0 + k] = np.median(rowbest)
    ci = np.zeros(rows, np.int64)
    hi = np.zeros(rows, np.int64)
    ci[-1], hi[-1] = divmod(int(np.argmax(acc)), H)
    for r in range(rows - 1, 0, -1):
        m = int(back[r, ci[r], hi[r]])
        ci[r - 1] = ci[r] - (m // 3 - 1)
        hi[r - 1] = hi[r] - (m % 3 - 1)
    # sub-column centre: parabola through the scores at c-1, c, c+1
    sub = np.zeros(rows)
    for r0 in range(0, rows, CH):
        r1 = min(rows, r0 + CH)
        S = _scores(p, edge, r0, r1, cidx, hcols, mcols, 0.8)
        rr = np.arange(r1 - r0)
        c0 = np.clip(ci[r0:r1], 1, C - 2)
        f0 = S[rr, c0 - 1, hi[r0:r1]]; f1 = S[rr, c0, hi[r0:r1]]; f2 = S[rr, c0 + 1, hi[r0:r1]]
        den = f0 - 2 * f1 + f2
        d = np.where(den < -1e-6, 0.5 * (f0 - f2) / np.where(den < -1e-6, den, -1), 0.0)
        sub[r0:r1] = (c0 - ci[r0:r1]) + np.clip(d, -0.5, 0.5)
    offset = off_m[ci] + sub * col_res
    conf = np.clip((best_score - second) / 0.5, 0.0, 1.0)
    conf = gaussian_filter1d(conf, sigma=max(1.0, 5.0 / row_step))
    return offset, hcols[hi] * col_res, conf


def _recentre(a, shift_cols):
    """resample every row so that column (cols-1)/2 lies `shift_cols[r]` columns away from the original centre"""
    rows, cols = a.shape[:2]
    mx = (np.arange(cols, dtype=np.float32)[None, :] + shift_cols.astype(np.float32)[:, None])
    my = np.repeat(np.arange(rows, dtype=np.float32)[:, None], cols, axis=1)
    return cv2.remap(a, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _shoulders(p, offset, half, col_res, width_m=1.5):
    """mean road-likeness just outside both detected edges; high on one side = junction / lay-by / wide apron"""
    rows, cols = p.shape
    m = max(2, int(round(width_m / col_res)))
    cs = np.concatenate([np.zeros((rows, 1), np.float32), np.cumsum(p, axis=1)], axis=1)
    c = offset / col_res + (cols - 1) / 2.0
    a = np.clip(np.round(c - half / col_res).astype(int), 0, cols)
    b = np.clip(np.round(c + half / col_res).astype(int) + 1, 0, cols)
    r = np.arange(rows)
    la = np.clip(a - m, 0, cols); rb = np.clip(b + m, 0, cols)
    left = (cs[r, a] - cs[r, la]) / np.maximum(a - la, 1)
    right = (cs[r, rb] - cs[r, b]) / np.maximum(rb - b, 1)
    return np.maximum(left, right)


def detect_road(img, rows, cols, row_step, col_res, max_offset=15.0, min_half=2.0, max_half=6.0,
                refine_window=5.0, offset_sigma_m=8.0, colour_window_m=1000.0):
    """steps 1–4 on one unrolled image (a whole route, or one section of a long route).
    Returns per-row arrays: offset (m, smoothed), off_raw (unsmoothed), half (m), rel (0–1), w_ok (0/1)."""
    I = np.frombuffer(bytes(img), np.uint8).reshape(rows, cols, 3)

    # 1. OpenCV: Lab colour space, light smoothing along the road (rows) only
    lab = cv2.cvtColor(I, cv2.COLOR_RGB2LAB).astype(np.float32)
    kr = max(3, int(round(2.0 / row_step)) * 2 + 1)
    lab = cv2.GaussianBlur(lab, (1, kr), sigmaX=0, sigmaY=max(0.5, 1.0 / row_step))

    # 2–4. two passes: (1) road colour learned under the map line as a running median over `colour_window_m`
    #       (long enough that a stretch where the map line misses the road does not set the colour, short enough
    #       to follow changes of asphalt, sun and shade along the route), searched up to `max_offset` m from the
    #       map line; (2) the image is re-centred on the road found in pass 1 and searched again
    #       (±refine_window m) with the colour learned locally (60 m) along that road
    a_c = 0.12 * (col_res / 0.25) / row_step   # per column of centre change per row (stiff: bridges trees/shadows)
    a_h = 0.08 * (col_res / 0.25) / row_step   # per column of width change per row
    w_pr = 0.002 * row_step                    # weak pull towards the search centre (per m²)
    c_mid = np.full(rows, (cols - 1) / 2.0)
    p, model, sig = _road_likelihood(lab, c_mid, col_res, row_step, colour_window_m)
    edge = _edges(lab, p, kr, row_step)
    off1, _, _ = _detect(p, edge, cols, col_res, row_step, max_offset, min_half, max_half, a_c, a_h, w_pr)
    off1 = gaussian_filter1d(off1, max(1.0, 4.0 / row_step))
    lab2 = _recentre(lab, off1 / col_res)
    del lab
    p, model, sig = _road_likelihood(lab2, c_mid, col_res, row_step, 60.0)
    edge = _edges(lab2, p, kr, row_step)
    del lab2
    off2, half_dp, conf = _detect(p, edge, cols, col_res, row_step, refine_window, min_half, max_half, a_c, a_h, w_pr)
    off_raw = off1 + off2
    # smoothing of the detected centre along the road (σ 8 m), weighted by confidence: detection noise is
    # correlated over ~10–20 m (tree shadows, paint, edge contrast), while the correction of the map line it
    # measures varies over the spacing of the map's vertices (tens of metres)
    wc = 0.2 + conf
    sg = max(1.0, offset_sigma_m / row_step)
    off2 = gaussian_filter1d(off2 * wc, sg) / gaussian_filter1d(wc, sg)

    # junctions, lay-bys and wide aprons: one edge of the road is not visible, so the image does not say where the
    # through-road's centre is. There the centre is bridged from the reliable stretches on both sides
    # (offset interpolated linearly along the road, i.e. the shape follows the map route between them).
    rr = np.arange(rows)
    open_side = gaussian_filter1d(_shoulders(p, off2, half_dp, col_res), max(1.0, 3.0 / row_step))
    rel = np.clip((0.45 - open_side) / 0.25, 0.0, 1.0) * np.clip(conf / 0.3, 0.0, 1.0)
    rel = gaussian_filter1d(rel, max(1.0, 3.0 / row_step))
    good = rel > 0.5
    if good.sum() >= 2:
        bridge = np.interp(rr, rr[good], off2[good])
        off2 = rel * off2 + (1.0 - rel) * bridge
    offset = off1 + off2

    # road width: asphalt plus painted lines (paint breaks the colour match, so short gaps are closed);
    # measured in the re-centred image around the pass-2 centre
    mask = (p > 0.2).astype(np.uint8)
    gap = max(3, int(round(1.25 / col_res)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, gap), np.uint8))
    mask = cv2.dilate(mask, np.ones((1, 3), np.uint8))
    cc = np.clip(np.round((off2 / col_res) + (cols - 1) / 2.0).astype(int), 0, cols - 1)
    left_e = np.zeros(rows); right_e = np.zeros(rows)
    for r in range(rows):
        m = mask[r]; c = cc[r]
        if not m[c]:
            for dc in (1, -1, 2, -2, 3, -3, 4, -4):
                if 0 <= c + dc < cols and m[c + dc]:
                    c = c + dc; break
        if not m[c]:
            left_e[r] = right_e[r] = np.nan
            continue
        a = c
        while a > 0 and m[a - 1]: a -= 1
        b = c
        while b < cols - 1 and m[b + 1]: b += 1
        left_e[r] = (a - (cols - 1) / 2.0 - 0.5) * col_res
        right_e[r] = (b - (cols - 1) / 2.0 + 0.5) * col_res
    wdet = right_e - left_e
    dp_w = 2 * half_dp
    wdet = np.where(np.isfinite(wdet) & (wdet <= 2 * max_half + 2.0), np.maximum(wdet, dp_w), dp_w)
    # the width is measured only where both edges are seen: not at junctions, and not where the band found is the
    # narrowest allowed (typically only the sunlit strip of a road whose other part lies in deep tree shadow);
    # elsewhere it is interpolated along the road from the measured stretches
    w_ok = good & (half_dp > min_half + 0.5 * col_res)
    if w_ok.sum() >= 2:
        wdet = np.where(w_ok, wdet, np.interp(rr, rr[w_ok], wdet[w_ok]))
    wdet = median_filter(wdet, size=max(3, int(round(30.0 / row_step)) | 1), mode='nearest')
    half = 0.5 * np.clip(wdet, 2 * min_half, 2 * max_half + 2.0)
    return dict(offset=offset.astype(np.float64), off_raw=off_raw.astype(np.float64), half=half.astype(np.float64),
                rel=rel.astype(np.float64), w_ok=w_ok.astype(np.float64), conf=conf.astype(np.float64),
                sigma_lab=[float(v) for v in sig])


def fit_centreline(X, Y, Nx, Ny, D, row_step, offset, off_raw, half, rel, w_ok=None, out_n=50000,
                   smooth_sigma=None, noise_scale_m=20.0, sigma_range=(0.25, 0.6)):
    """step 5: centre points → smoothing B-spline (scipy splprep) → out_n equidistant points"""
    X, Y, Nx, Ny, D, offset, off_raw, half, rel = (np.asarray(v, np.float64)
                                                   for v in (X, Y, Nx, Ny, D, offset, off_raw, half, rel))
    rows = len(X)
    cx = X + offset * Nx
    cy = Y + offset * Ny
    w = 0.3 + rel
    w = w / w.mean()
    keep = np.r_[True, np.hypot(np.diff(cx), np.diff(cy)) > 1e-3]
    cx, cy, w = cx[keep], cy[keep], w[keep]
    rid = np.nonzero(keep)[0].astype(np.float64)
    # smoothing factor from the measured detection noise: robust spread (MAD) of the detected centre around its
    # 20 m running mean, before smoothing
    if smooth_sigma is None:
        hp = off_raw - gaussian_filter1d(off_raw, max(1.0, noise_scale_m / row_step))
        smooth_sigma = float(np.clip(1.4826 * np.median(np.abs(hp - np.median(hp))), *sigma_range))
    s_fac = len(cx) * smooth_sigma ** 2
    tck, u = splprep([cx, cy], w=w, s=s_fac, k=3)
    ud = np.linspace(0.0, 1.0, max(out_n * 2, 20000))
    xd, yd = splev(ud, tck)
    cl = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xd), np.diff(yd)))])
    target = np.linspace(0.0, cl[-1], out_n)
    uo = np.interp(target, cl, ud)
    xo, yo = splev(uo, tck)
    dx, dy = splev(uo, tck, der=1)
    ddx, ddy = splev(uo, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    # map each output point back to a route station (for elevation / width lookups)
    rsrc = np.interp(uo, u, rid)
    wid = gaussian_filter1d(2.0 * half, sigma=max(1.0, 15.0 / row_step))
    width_o = np.interp(rsrc, np.arange(rows), wid)
    D_o = np.interp(rsrc, np.arange(rows), D)
    conf_o = np.interp(rsrc, np.arange(rows), rel)
    resid = np.hypot(np.interp(u, uo, xo) - cx, np.interp(u, uo, yo) - cy)
    stats = dict(
        rows=int(rows), length=float(cl[-1]), out_n=int(out_n), spacing=float(cl[-1] / max(1, out_n - 1)),
        shift_mean=float(np.mean(np.abs(offset))), shift_p95=float(np.percentile(np.abs(offset), 95)),
        shift_max=float(np.max(np.abs(offset))),
        width_median=float(np.median(2 * half)), bridged_frac=float(np.mean(rel < 0.5)),
        width_measured_frac=float(np.mean(np.asarray(w_ok) > 0.5)) if w_ok is not None else None,
        spline_resid_rms=float(np.sqrt(np.mean(resid ** 2))), kappa_max=float(np.max(np.abs(kappa))),
        smooth_sigma=float(smooth_sigma),
    )
    return dict(x=np.asarray(xo, np.float64), y=np.asarray(yo, np.float64), width=width_o.astype(np.float32),
                D=D_o.astype(np.float64), conf=conf_o.astype(np.float32), kappa=kappa.astype(np.float32),
                stats=stats)


def fit_road(img, rows, cols, row_step, col_res, X, Y, Nx, Ny, D, out_n=50000,
             max_offset=15.0, min_half=2.0, max_half=6.0, smooth_sigma=None, refine_window=5.0,
             offset_sigma_m=8.0, noise_scale_m=20.0, sigma_range=(0.25, 0.6)):
    """whole pipeline on one unrolled image"""
    det = detect_road(img, rows, cols, row_step, col_res, max_offset, min_half, max_half, refine_window,
                      offset_sigma_m)
    res = fit_centreline(X, Y, Nx, Ny, D, row_step, det['offset'], det['off_raw'], det['half'], det['rel'],
                         det['w_ok'], out_n, smooth_sigma, noise_scale_m, sigma_range)
    res['stats']['sigma_lab'] = det['sigma_lab']
    res['stats']['conf_low_frac'] = float(np.mean(det['conf'] < 0.25))
    res.update(offset=det['offset'].astype(np.float32), half=det['half'].astype(np.float32),
               row_conf=det['rel'].astype(np.float32))
    return res
