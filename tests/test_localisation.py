"""Self-tests with synthetic heatmaps; run from the repository root: python tests/test_localisation.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np
from localisation import *
# 1. mapping round trip on square and non-square images: a peak painted at a known place in the 224 frame
for (W, H) in [(512, 512), (201, 251), (630, 590)]:
    s, ox, oy = pad_offset(W, H)
    # choose a target pixel in the original image, find its location in the 224 frame
    tx, ty = int(W * 0.7), int(H * 0.3)
    px, py = int((tx + ox) / s * 224), int((ty + oy) / s * 224)
    cam = np.zeros((224, 224), np.float32); cam[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3] = 1
    m, pad = to_original(cam, W, H)
    iy, ix = np.unravel_index(m.argmax(), m.shape)
    err = max(abs(ix - tx), abs(iy - ty)); tol = s / 224 * 3
    print(f'{W}x{H}: target ({tx},{ty}) -> peak ({ix},{iy}), error {err}px, tolerance {tol:.1f}px, pad share {pad:.3f}')
    assert err <= tol and m.shape == (H, W) and 0 <= m.min() and m.max() == 1
# 2. a heatmap that is only on padding gives a nearly flat crop and a large pad share
cam = np.zeros((224, 224), np.float32); cam[:, :5] = 1
m, pad = to_original(cam, 201, 251); print('padding-only heat: pad share', round(pad, 3)); assert pad > .9
# 3. box mask
boxes = [[0, .5, .5, .2, .1], [3, .1, .1, .1, .1]]
mk, n = box_mask(boxes, 200, 100, class_id=0); assert n == 1 and abs(mk.mean() - .02) < .005, (n, mk.mean())
mk_all, n2 = box_mask(boxes, 200, 100); assert n2 == 2
# 4. metrics: perfect, disjoint and uniform heatmaps
H, W = 100, 200; mask, _ = box_mask([[0, .5, .5, .2, .2]], W, H)
heat = np.zeros((H, W)); heat[mask] = 1; r = metrics_for_map(heat, mask); assert r['hit'] and abs(r['iou_0.5'] - 1) < 1e-9 and r['enrichment'] > 4
heat2 = np.zeros((H, W)); heat2[:10, :10] = 1; r2 = metrics_for_map(heat2, mask); assert not r2['hit'] and r2['iou_0.5'] == 0
heat3 = np.ones((H, W)); r3 = metrics_for_map(heat3, mask); assert abs(r3['enrichment'] - 1) < 1e-9
# 5. random heatmaps hit boxes at about the box-area fraction
rng = np.random.default_rng(0); hits = [metrics_for_map(rng.random((H, W)), mask)['hit'] for _ in range(4000)]
print('random heatmap hit rate', np.mean(hits), 'expected', mask.mean()); assert abs(np.mean(hits) - mask.mean()) < .012
# 6. centre prior is symmetric and peaks in the middle
c = centre_prior_224(); assert np.unravel_index(c.argmax(), c.shape) in [(111, 111), (111, 112), (112, 111), (112, 112)]
# 7. stats helpers
lo, hi = wilson(50, 100); assert .40 < lo < .41 and .59 < hi < .60
print('boot', boot_ci(np.arange(100)), 'spearman', rank_corr(np.arange(50), np.arange(50)))
print('ALL LOCALISATION TESTS PASSED')
