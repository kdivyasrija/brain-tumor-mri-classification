"""Stage 8: Grad-CAM and comparison with the annotated bounding boxes (final model, test set).

Usage: python gradcam_analysis.py --results <final_results> --data-root <Train/Val folder> --ckpt <R18-C_best.pt> --out <dir> [--src <folder with gradcam.py, localisation.py>]

What is measured (all computed here, nothing is typed in):
  * sanity checks: checkpoint reproduces the saved test predictions; hooked gradients equal the analytic ones;
    heatmaps change when the weights are randomised
  * localisation of Grad-CAM peaks and hot regions against the annotated boxes (tumour classes only), with baselines
What is NOT claimed: that the heatmaps show what the network 'uses', or that the model localises tumours. Boxes are an
imperfect reference (Stage 1 found annotation oddities) and the layer4 map is only 7 x 7.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

CL = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']; TUMOUR = {'Glioma': 0, 'Meningioma': 1, 'Pituitary': 3}
COL = {'Grad-CAM (layer4)': '#2a78d6', 'Grad-CAM (layer3)': '#1baf7a', 'Centre prior': '#8a8985', 'Permuted heatmap': '#b9b8b3', 'Random point': '#e4e3df'}
INK = '#0b0b0b'


def main():
    ap = argparse.ArgumentParser()
    for k in ['results', 'data_root', 'ckpt', 'out']: ap.add_argument('--' + k.replace('_', '-'), required=True)
    ap.add_argument('--src', default=str(Path(__file__).resolve().parent.parent / 'src')); ap.add_argument('--device', default='cuda'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-random-check', type=int, default=60)
    a = ap.parse_args(); sys.path.insert(0, a.src)
    import localisation as L
    import gradcam as G
    res, root, out = Path(a.results), Path(a.data_root), Path(a.out)
    for d in ['tables', 'figures']: (out / d).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
    sp = pd.read_csv(res / 'metrics' / 'splits.csv'); te = sp[sp.split == 'test'].reset_index(drop=True)
    paths = [str(root / r) for r in te.rel]; N = len(te); sanity = {}
    ex, ck = G.Explainer.from_checkpoint(a.ckpt, a.device)
    sanity['checkpoint'] = dict(epoch=int(ck.get('epoch', -1)), monitor=ck.get('monitor'), monitor_value=float(ck.get('monitor_value', float('nan'))))

    # ---- 1. forward pass: does this checkpoint reproduce a saved run? ------------------------------------------------
    base = ex.forward_cam(paths, 'layer4'); P = base['probs']; pred = base['preds']
    match = {}
    for cand in ['R18-C', 'R18-C-orig']:
        f = res / 'predictions' / f'{cand}_test_predictions.csv'
        if not f.exists(): continue
        s = pd.read_csv(f).set_index('rel').loc[te.rel]; sp_probs = s[[f'prob_{c}' for c in CL]].values
        match[cand] = dict(label_agreement=float((s.y_pred.values == pred).mean()), max_abs_prob_diff=float(np.abs(sp_probs - P).max()),
                           n_label_disagreements=int((s.y_pred.values != pred).sum()))
    sanity['reproduces_saved_predictions'] = match
    best = min(match, key=lambda k: match[k]['n_label_disagreements']); sanity['matched_run'] = best
    assert match[best]['label_agreement'] >= 0.99, f'checkpoint does not reproduce a saved run: {match}'
    saved = pd.read_csv(res / 'predictions' / f'{best}_test_predictions.csv').set_index('rel').loc[te.rel]
    te['pred'] = pred; te['prob_pred'] = P.max(1); te['correct'] = te.label_idx.values == pred
    print('Checkpoint matches saved run', best, match[best])

    # ---- 2. hook correctness -----------------------------------------------------------------------------------------
    sanity['analytic_gradient_check'] = chk = ex.analytic_check(paths[:16])
    assert chk['max_abs_alpha_error'] < 1e-3 * max(chk['alpha_scale'], 1e-9) + 1e-7 and chk['max_abs_logit_reconstruction_error'] < 1e-3, chk
    print('Analytic gradient check passed:', chk)

    # ---- 3. heatmaps --------------------------------------------------------------------------------------------------
    cam4 = base['cams']; cam3 = ex.forward_cam(paths, 'layer3')['cams']
    err_idx = np.where(~te.correct.values)[0]
    cam4_true = ex.forward_cam([paths[i] for i in err_idx], 'layer4', targets=te.label_idx.values[err_idx])['cams'] if len(err_idx) else np.zeros((0, 224, 224), np.float32)
    flat4 = int((cam4.reshape(N, -1).max(1) == 0).sum()); sanity['images_with_all_zero_layer4_cam'] = flat4

    # ---- 4. localisation ----------------------------------------------------------------------------------------------
    rng = np.random.default_rng(a.seed); rows = []; pad_shares = []
    ev = te[te.label.isin(TUMOUR)].copy(); ev['boxes'] = ev.boxes_json.map(json.loads)
    n_no_match = 0; centre = L.centre_prior_224(); by_class_idx = {c: ev.index[ev.label == c].values for c in TUMOUR}
    perm = {r: {} for r in range(5)}
    for r in range(5):  # five independent draws of a heatmap from ANOTHER test image of the same class
        rr = np.random.default_rng(a.seed + 1 + r)
        for c, idx in by_class_idx.items():
            for i in idx:
                others = idx[idx != i]; perm[r][i] = int(rr.choice(others))
    for i, row in ev.iterrows():
        W, H = int(row.width), int(row.height)
        mask, used = L.box_mask(row.boxes, W, H, class_id=TUMOUR[row.label])
        if used == 0: n_no_match += 1; continue
        rec = dict(rel=row.rel, label=row.label, pred=CL[row.pred], correct=bool(row.correct), prob_pred=row.prob_pred, W=W, H=H, n_boxes_used=used)
        h4, ps = L.to_original(cam4[i], W, H); rec['pad_share_l4'] = ps
        for name, heat in [('l4', h4), ('l3', L.to_original(cam3[i], W, H)[0]), ('centre', L.to_original(centre, W, H)[0])]:
            for k, v in L.metrics_for_map(heat, mask).items(): rec[f'{name}_{k}'] = v
        for r in range(5):
            m = L.metrics_for_map(L.to_original(cam4[perm[r][i]], W, H)[0], mask)
            for k in ['hit', 'iou_0.5', 'enrichment']: rec[f'perm{r}_{k}'] = m[k]
        rec['random_point_hit_expected'] = float(mask.mean())
        rows.append(rec)
    D = pd.DataFrame(rows); D.to_csv(out / 'tables' / 'gradcam_per_image_test.csv', index=False)
    sanity['tumour_test_images'] = int(len(ev)); sanity['tumour_images_without_matching_class_box'] = int(n_no_match); sanity['evaluated_images'] = int(len(D))

    def summ(d, tag):
        n = len(d); out_rows = []
        for name, pre in [('Grad-CAM (layer4)', 'l4'), ('Grad-CAM (layer3)', 'l3'), ('Centre prior', 'centre')]:
            k = int(d[f'{pre}_hit'].sum()); lo, hi = L.wilson(k, n)
            out_rows.append(dict(group=tag, method=name, n=n, pointing_hits=k, pointing_rate=k / n, pointing_ci_low=lo, pointing_ci_high=hi,
                                 iou_0_5_mean=d[f'{pre}_iou_0.5'].mean(), iou_0_5_ci_low=L.boot_ci(d[f'{pre}_iou_0.5'])[0], iou_0_5_ci_high=L.boot_ci(d[f'{pre}_iou_0.5'])[1],
                                 iou_0_3_mean=d[f'{pre}_iou_0.3'].mean(), iou_0_7_mean=d[f'{pre}_iou_0.7'].mean(),
                                 enrichment_median=d[f'{pre}_enrichment'].median(), enrichment_mean=d[f'{pre}_enrichment'].mean(),
                                 enrichment_ci_low=L.boot_ci(d[f'{pre}_enrichment'])[0], enrichment_ci_high=L.boot_ci(d[f'{pre}_enrichment'])[1]))
        ph = [d[f'perm{r}_hit'].mean() for r in range(5)]; pi = [d[f'perm{r}_iou_0.5'].mean() for r in range(5)]; pe = [d[f'perm{r}_enrichment'].mean() for r in range(5)]
        out_rows.append(dict(group=tag, method='Permuted heatmap', n=n, pointing_rate=np.mean(ph), pointing_ci_low=min(ph), pointing_ci_high=max(ph),
                             iou_0_5_mean=np.mean(pi), iou_0_5_ci_low=min(pi), iou_0_5_ci_high=max(pi), enrichment_mean=np.mean(pe), enrichment_ci_low=min(pe), enrichment_ci_high=max(pe)))
        out_rows.append(dict(group=tag, method='Random point', n=n, pointing_rate=d.random_point_hit_expected.mean()))
        return out_rows
    S = pd.DataFrame(summ(D, 'all evaluated tumour images'))
    for c in TUMOUR: S = pd.concat([S, pd.DataFrame(summ(D[D.label == c], c))])
    S = pd.concat([S, pd.DataFrame(summ(D[D.correct], 'correctly classified')), pd.DataFrame(summ(D[~D.correct], 'misclassified'))] if (~D.correct).any() else [S, pd.DataFrame(summ(D[D.correct], 'correctly classified'))])
    D['size_tercile'] = pd.qcut(D.l4_box_area_frac, 3, labels=['small boxes', 'medium boxes', 'large boxes'])
    for t, d in D.groupby('size_tercile', observed=True): S = pd.concat([S, pd.DataFrame(summ(d, f'{t} (area {d.l4_box_area_frac.min():.3f} to {d.l4_box_area_frac.max():.3f} of image)'))])
    S.to_csv(out / 'tables' / 'gradcam_localisation_summary.csv', index=False)
    print(S[S.group == 'all evaluated tumour images'][['method', 'n', 'pointing_rate', 'pointing_ci_low', 'pointing_ci_high', 'iou_0_5_mean', 'enrichment_median']].round(3).to_string(index=False))

    # ---- 5. weight-randomisation sanity check -------------------------------------------------------------------------
    ev_pos = ev.index.values; sel = rng.choice(ev_pos, size=min(a.n_random_check, len(ev_pos)), replace=False)
    sel_paths = [paths[i] for i in sel]; trained = cam4[sel]; rand_rows = []; rand_cams = {}
    for mode in ['fc', 'layer4+fc', 'all']:
        rx = ex.randomised(mode, seed=a.seed); r_cam = rx.forward_cam(sel_paths, 'layer4', targets=pred[sel])['cams']; rand_cams[mode] = r_cam
        rho = [L.rank_corr(trained[j], r_cam[j]) if r_cam[j].max() > 0 and trained[j].max() > 0 else float('nan') for j in range(len(sel))]
        hit = []
        for j, i in enumerate(sel):
            row = te.loc[i]; mask, used = L.box_mask(json.loads(row.boxes_json), int(row.width), int(row.height), class_id=TUMOUR[row.label])
            if used: hit.append(L.metrics_for_map(L.to_original(r_cam[j], int(row.width), int(row.height))[0], mask)['hit'])
        rand_rows.append(dict(randomised=mode, n_images=len(sel), mean_spearman_with_trained_map=float(np.nanmean(rho)), median_spearman=float(np.nanmedian(rho)),
                              pointing_rate_randomised=float(np.mean(hit)), all_zero_maps=int(sum(r_cam[j].max() == 0 for j in range(len(sel))))))
    R = pd.DataFrame(rand_rows); R.to_csv(out / 'tables' / 'gradcam_randomisation_check.csv', index=False); sanity['randomisation_check'] = rand_rows
    print(R.round(3).to_string(index=False))

    # ---- 6. figures ---------------------------------------------------------------------------------------------------
    from PIL import Image
    def show(ax, i, cam, title, boxes=True):
        row = te.loc[i]; img = Image.open(root / row.rel).convert('L'); W, H = img.size
        heat = L.to_original(cam, W, H)[0]
        ax.imshow(img, cmap='gray', vmin=0, vmax=255); ax.imshow(heat, cmap='magma', alpha=0.45, vmin=0, vmax=1)
        if heat.max() > 0: ax.contour(heat, levels=[0.5], colors='white', linewidths=0.8, linestyles='dotted')
        if boxes:
            mask, _ = L.box_mask(json.loads(row.boxes_json), W, H, class_id=TUMOUR.get(row.label, -1))
            if mask.any(): ax.contour(mask.astype(float), levels=[0.5], colors='#7fe3ff', linewidths=1.4)
        ax.set_title(title, fontsize=8, loc='left'); ax.axis('off')
    def grid(items, path, ncols=4, suptitle=''):
        nr = int(np.ceil(len(items) / ncols)); fig, axs = plt.subplots(nr, ncols, figsize=(3.2 * ncols, 3.5 * nr)); axs = np.atleast_2d(axs)
        for ax in axs.ravel(): ax.axis('off')
        for ax, (i, cam, t) in zip(axs.ravel(), items): show(ax, i, cam, t)
        fig.tight_layout(h_pad=2.2, rect=[0, 0, 1, 0.97]); fig.suptitle(suptitle, fontsize=9, x=0.01, y=0.995, ha='left'); fig.savefig(path, dpi=170, bbox_inches='tight'); plt.close(fig)
    cap = 'cyan outline = annotated box; dotted white = 50% contour of the heatmap; heat is min-max normalised per image'
    ok = D[D.correct]; items = []
    for c in TUMOUR:
        pick = ok[ok.label == c].sample(min(4, (ok.label == c).sum()), random_state=a.seed)
        for _, r in pick.iterrows():
            i = int(te.index[te.rel == r.rel][0]); items.append((i, cam4[i], f'{r.label}: correct, p={r.prob_pred:.2f}\npeak {"inside" if r.l4_hit else "outside"} box'))
    grid(items, out / 'figures' / 'gradcam_random_correct_examples.png', suptitle='Random correctly classified test images (4 per tumour class, fixed seed). ' + cap)
    items = []
    for j, i in enumerate(err_idx):
        row = te.loc[i]; items.append((int(i), cam4[i], f'TRUE {row.label} -> PRED {CL[row.pred]} (p={row.prob_pred:.2f})\nmap for predicted class'))
        if row.label in TUMOUR: items.append((int(i), cam4_true[j], f'same image, map for TRUE class {row.label}'))
    if items: grid(items, out / 'figures' / 'gradcam_misclassified_examples.png', suptitle='All misclassified test images of the final model. ' + cap)
    # bar chart vs baselines
    A = S[S.group == 'all evaluated tumour images'].set_index('method'); order = ['Grad-CAM (layer4)', 'Grad-CAM (layer3)', 'Centre prior', 'Permuted heatmap', 'Random point']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    v = A.loc[order, 'pointing_rate'].values * 100; lo = A.loc[order, 'pointing_ci_low'].values * 100; hi = A.loc[order, 'pointing_ci_high'].values * 100
    for k, m in enumerate(order):
        ax[0].barh(k, v[k], color=COL[m]); 
        if not np.isnan(lo[k]): ax[0].plot([lo[k], hi[k]], [k, k], color=INK, lw=1.2)
        ax[0].text(max(v[k], hi[k] if not np.isnan(hi[k]) else 0) + 1, k, f'{v[k]:.1f}%', va='center')
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order); ax[0].invert_yaxis(); ax[0].set_xlim(0, 105)
    ax[0].set_xlabel('Pointing game: peak of heatmap inside an annotated box (%)'); ax[0].set_title(f'n = {len(D)} tumour test images', loc='left')
    iv = A.loc[order[:4], 'iou_0_5_mean'].values; il = A.loc[order[:4], 'iou_0_5_ci_low'].values; ih = A.loc[order[:4], 'iou_0_5_ci_high'].values
    for k, m in enumerate(order[:4]):
        ax[1].barh(k, iv[k], color=COL[m]); ax[1].plot([il[k], ih[k]], [k, k], color=INK, lw=1.2); ax[1].text(ih[k] + .005, k, f'{iv[k]:.3f}', va='center')
    ax[1].set_yticks(range(4)); ax[1].set_yticklabels(order[:4]); ax[1].invert_yaxis(); ax[1].set_xlabel('Mean IoU of hot region (>= 50% of peak) with box'); ax[1].set_xlim(0, max(ih.max() * 1.25, .1))
    ax[1].set_title('whiskers: 95% CI (Wilson / bootstrap); permuted: range of 5 draws', loc='left', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'gradcam_localisation_vs_baselines.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    # by class and size
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for axx, col, ttl in [(ax[0], 'label', 'By tumour class'), (ax[1], 'size_tercile', 'By box size (terciles of box area)')]:
        gs = D.groupby(col, observed=True); xs = np.arange(len(gs)); w = .38
        for o, (pre, nm) in enumerate([('l4', 'Grad-CAM (layer4)'), ('centre', 'Centre prior')]):
            rates = [g[f'{pre}_hit'].mean() * 100 for _, g in gs]; axx.bar(xs + (o - .5) * w, rates, w, color=COL[nm], label=nm)
            for x_, r_, (_, g) in zip(xs + (o - .5) * w, rates, gs): axx.text(x_, r_ + 1, f'{r_:.0f}%', ha='center', fontsize=8)
        axx.set_xticks(xs); axx.set_xticklabels([f'{k}\n(n={len(g)})' for k, g in gs]); axx.set_ylim(0, 112); axx.set_ylabel('Pointing hit rate (%)'); axx.set_title(ttl, loc='left')
    ax[0].legend(loc='lower left'); fig.tight_layout(); fig.savefig(out / 'figures' / 'gradcam_localisation_by_class_and_size.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    # randomisation figure
    show_idx = sel[:3]; fig, axs = plt.subplots(len(show_idx), 4, figsize=(12.5, 3.3 * len(show_idx))); axs = np.atleast_2d(axs)
    for r_, i in enumerate(show_idx):
        show(axs[r_, 0], int(i), cam4[i], f'trained model ({te.loc[i].label})')
        for c_, mode in enumerate(['fc', 'layer4+fc', 'all']): show(axs[r_, c_ + 1], int(i), rand_cams[mode][list(sel).index(i)], f'randomised: {mode}')
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.suptitle('Sanity check: heatmaps should change when weights are randomised', fontsize=10, x=0.01, y=0.995, ha='left'); fig.savefig(out / 'figures' / 'gradcam_randomisation_check.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    sanity['mean_heat_share_on_padding_layer4'] = float(np.nanmean(D.pad_share_l4)); sanity['note'] = 'Boxes used: only boxes whose class id equals the image class. No Tumor images are excluded (their boxes cover most of the brain).'
    json.dump(sanity, open(out / 'gradcam_sanity_checks.json', 'w'), indent=1, default=float)
    print('\nDONE. Files in', out)


if __name__ == '__main__':
    main()
