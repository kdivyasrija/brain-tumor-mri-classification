"""Geometry and localisation metrics for Grad-CAM heatmaps. NumPy/PIL/SciPy only (no PyTorch).

Coordinate conventions
  * An original image is W x H pixels. Preprocessing pads it with black to an s x s square, s = max(W, H),
    pasting it at (ox, oy) = ((s - W) // 2, (s - H) // 2), then resizes to 224 x 224 (same as PadToSquare in Stage 2).
  * A heatmap computed on the 224 x 224 network input is mapped back by: resize to s x s -> crop the original
    area [oy:oy+H, ox:ox+W]. After the crop it is min-max normalised inside the original image area.
  * Boxes are YOLO lines [class, cx, cy, w, h], normalised to the ORIGINAL image size (verified in Stage 1).
"""
import numpy as np
from PIL import Image
from scipy.stats import spearmanr


def pad_offset(W, H):
    s = max(W, H)
    return s, (s - W) // 2, (s - H) // 2


def to_original(cam224, W, H):
    """cam224: (224,224) float array -> (H, W) float array in original pixel coordinates, min-max normalised.
    Also returns the share of the heatmap mass that falls on the black padding (before cropping)."""
    s, ox, oy = pad_offset(W, H)
    big = np.asarray(Image.fromarray(cam224.astype(np.float32), mode='F').resize((s, s), Image.BILINEAR), dtype=np.float64)
    big = np.clip(big, 0, None)
    tot = big.sum()
    crop = big[oy:oy + H, ox:ox + W]
    pad_share = float(1 - crop.sum() / tot) if tot > 0 else float('nan')
    lo, hi = crop.min(), crop.max()
    crop = (crop - lo) / (hi - lo) if hi > lo else np.zeros_like(crop)
    return crop, pad_share


def box_mask(boxes, W, H, class_id=None):
    """Union mask of the boxes (optionally only those with the given class id). Returns (mask, n_boxes_used)."""
    m = np.zeros((H, W), dtype=bool); used = 0
    for c, cx, cy, bw, bh in boxes:
        if class_id is not None and int(c) != class_id: continue
        x0 = int(np.floor((cx - bw / 2) * W)); x1 = int(np.ceil((cx + bw / 2) * W))
        y0 = int(np.floor((cy - bh / 2) * H)); y1 = int(np.ceil((cy + bh / 2) * H))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 > x0 and y1 > y0: m[y0:y1, x0:x1] = True; used += 1
    return m, used


def centre_prior_224(sigma_frac=0.25, size=224):
    """Fixed Gaussian centred on the network input. Baseline that uses no image information."""
    y, x = np.mgrid[0:size, 0:size]; c = (size - 1) / 2; s = sigma_frac * size
    return np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * s * s)).astype(np.float32)


def metrics_for_map(heat, mask, thresholds=(0.3, 0.5, 0.7)):
    """heat: (H,W) in [0,1]; mask: (H,W) bool ground-truth box union. All values are computed here."""
    H, W = heat.shape
    iy, ix = np.unravel_index(int(np.argmax(heat)), heat.shape)
    out = {'hit': bool(mask[iy, ix]), 'peak_x': int(ix), 'peak_y': int(iy), 'box_area_frac': float(mask.mean())}
    for t in thresholds:
        b = heat >= t * heat.max() if heat.max() > 0 else np.zeros_like(mask)
        inter = np.logical_and(b, mask).sum(); union = np.logical_or(b, mask).sum()
        out[f'iou_{t}'] = float(inter / union) if union else float('nan')
        out[f'hot_area_frac_{t}'] = float(b.mean())
    tot = heat.sum()
    out['enrichment'] = float((heat[mask].sum() / tot) / mask.mean()) if tot > 0 and mask.any() else float('nan')
    return out


def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def boot_ci(x, stat=np.mean, n=2000, seed=0):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) == 0: return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    v = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def rank_corr(a, b):
    """Spearman correlation between two heatmaps of the same shape."""
    r = spearmanr(np.ravel(a), np.ravel(b)).correlation
    return float(r)
