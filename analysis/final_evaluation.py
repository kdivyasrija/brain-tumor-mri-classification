"""Stage 7: final evaluation and error analysis.

Reads ONLY saved results (prediction files with class probabilities, training histories, the experiment table,
the locked split table) plus the MRI images for the example figures. It needs no GPU and no PyTorch.
Every number written to disk is computed here from those files.

Usage:
    python final_evaluation.py --results final_results --data-root <folder containing Train/ and Val/> --src <project src folder> --out stage7_results
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.stats import fisher_exact
from sklearn.metrics import roc_curve, auc, classification_report

CL = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
# Class colours: first four slots of the validated categorical palette (blue, orange, aqua, yellow).
# Aqua and yellow are below 3:1 contrast on white, so every figure also uses direct labels / line styles / a CSV table.
CLASS_COL = {'Glioma': '#2a78d6', 'Meningioma': '#eb6834', 'No Tumor': '#1baf7a', 'Pituitary': '#eda100'}
FAMILY_COL = {'CNN from scratch': '#8a8985', 'ResNet18': '#2a78d6', 'ResNet50': '#eb6834'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
FULL = {'ResNet18': ['R18-C', 'R18-C-s43', 'R18-C-s44'], 'ResNet50': ['R50-C', 'F-R50-s43', 'F-R50-s44']}
FINAL = 'R18-C'
CONFIGS = [('CNN from scratch', 'CNN from scratch', ['B1']),
           ('CNN, weighted loss', 'CNN from scratch', ['B2']),
           ('ResNet18, frozen backbone', 'ResNet18', ['R18-A']),
           ('ResNet18, layer4 + fc', 'ResNet18', ['R18-B']),
           ('ResNet18, full fine-tuning', 'ResNet18', FULL['ResNet18']),
           ('ResNet50, frozen backbone', 'ResNet50', ['R50-A']),
           ('ResNet50, layer4 + fc', 'ResNet50', ['R50-B']),
           ('ResNet50, full fine-tuning', 'ResNet50', FULL['ResNet50'])]
TUNING = ['T1-lr1e-5', 'T2-lr1e-3', 'T3-bs16', 'T4-adamw', 'T5-dropout', 'T6-weighted', 'T7-hflip']


# ----------------------------------------------------------------------------------------------- helpers
def style():
    plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': .8, 'axes.axisbelow': True, 'figure.facecolor': 'white',
                         'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'legend.frameon': False})


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_pred(res, exp, split):
    p = pd.read_csv(res / 'predictions' / f'{exp}_{split}_predictions.csv').set_index('rel').sort_index()
    y = p.true_label.map({c: i for i, c in enumerate(CL)}).values
    P = p[[f'prob_{c}' for c in CL]].values
    return p, y, P


def savefig(fig, path):
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)


def heatmap(ax, cm, title):
    cm = np.asarray(cm); norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            col = 'white' if norm[i, j] > .55 else INK
            ax.text(j, i, f'{cm[i, j]}\n{norm[i, j] * 100:.1f}%', ha='center', va='center', color=col, fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(CL, rotation=20, ha='right'); ax.set_yticks(range(4)); ax.set_yticklabels(CL)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title(title, fontsize=10); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)


def draw_img(ax, root, rel, row, show_box, title):
    im = Image.open(root / rel).convert('L'); W, H = im.size
    ax.imshow(im, cmap='gray', vmin=0, vmax=255); ax.axis('off')
    if show_box:
        for b in json.loads(row.boxes_json):
            _, cx, cy, w, h = b
            ax.add_patch(Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False, edgecolor='#ffd23f', lw=1.3, ls='--'))
    ax.set_title(title, fontsize=7.8, color=INK, pad=3)


def source_style(stem):
    """Filename pattern as a rough proxy for the original source of an image (dataset appears to merge several collections)."""
    if re.match(r'^(Tr|Te)-', stem): return 'Tr-/Te- prefixed'
    if re.match(r'^gg\s*\(', stem): return 'gg (n)'
    if re.match(r'^p\s*\(', stem): return 'p (n)'
    if re.match(r'^m\d?\s*\(', stem): return 'm (n) / m1-3 (n)'
    if re.match(r'^image\s*\(', stem): return 'image (n)'
    if re.match(r'^\d+$', stem): return 'number only'
    return 'other'


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True); ap.add_argument('--data-root', required=True)
    ap.add_argument('--src', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res, root, out = Path(a.results), Path(a.data_root), Path(a.out)
    (out / 'tables').mkdir(parents=True, exist_ok=True); (out / 'figures').mkdir(exist_ok=True)
    sys.path.insert(0, a.src)
    from evaluate import compute_metrics, bootstrap_ci   # pure numpy/sklearn functions from the project
    import visualization as vz
    style()
    table = pd.read_csv(res / 'metrics' / 'experiment_table_master.csv').set_index('experiment_id')
    splits = pd.read_csv(res / 'metrics' / 'splits.csv').set_index('rel')
    S = {}   # summary numbers for the report

    # ---- 1. metrics for every run, recomputed from the saved probabilities
    M = {}
    rows = []
    for e in table.index:
        r = {'experiment_id': e}
        for split in ['val', 'test']:
            p, y, P = load_pred(res, e, split)
            m = compute_metrics(y, P, CL); M[(e, split)] = (m, y, P)
            s = m['summary']
            r.update({f'{split}_{k}': s[k] for k in ['accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'roc_auc_ovr_macro']})
        rows.append(r)
    allruns = pd.DataFrame(rows).set_index('experiment_id')
    for c, tc in [('val_accuracy', 'val_accuracy'), ('test_accuracy', 'test_accuracy'), ('val_f1_macro', 'val_macro_f1'), ('test_f1_macro', 'test_macro_f1')]:
        assert np.allclose(allruns[c], table.loc[allruns.index, tc], atol=1e-6), f'recomputed {c} differs from the experiment table'
    allruns = allruns.join(table[['model', 'finetune_strategy', 'seed', 'best_epoch', 'epochs_run', 'trainable_params', 'train_time_s']])
    allruns.round(4).to_csv(out / 'tables' / 'all_runs_metrics.csv')
    S['recomputed_metrics_match_experiment_table'] = True

    # ---- 2. model comparison table (the table requested in the project brief)
    def agg(ids, split, col):
        v = allruns.loc[ids, f'{split}_{col}'].values
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan)
    cols = [('accuracy', 'Accuracy'), ('precision_macro', 'Precision (macro)'), ('recall_macro', 'Recall (macro)'), ('f1_macro', 'F1 (macro)'),
            ('roc_auc_ovr_macro', 'ROC-AUC (OvR macro)'), ('balanced_accuracy', 'Balanced accuracy')]
    spec = [('CNN baseline (B1), single run', ['B1']), ('ResNet18, full fine-tuning, mean of 3 seeds', FULL['ResNet18']),
            ('ResNet50, full fine-tuning, mean of 3 seeds', FULL['ResNet50']),
            ('Final model: ResNet18 full fine-tuning, seed-42 run (R18-C)', [FINAL])]
    num, txt = [], []
    for name, ids in spec:
        rn, rt = {'model': name, 'runs': ' + '.join(ids)}, {'Model': name}
        for k, lab in cols:
            mu, sd = agg(ids, 'test', k); rn[lab + ' mean'] = mu; rn[lab + ' std'] = sd
            rt[lab] = f'{mu:.4f} ± {sd:.4f}' if len(ids) > 1 else f'{mu:.4f}'
        num.append(rn); txt.append(rt)
    pd.DataFrame(num).round(4).to_csv(out / 'tables' / 'model_comparison_numeric.csv', index=False)
    pd.DataFrame(txt).to_csv(out / 'tables' / 'model_comparison_table.csv', index=False)
    S['model_comparison'] = txt
    ext = []
    for label, fam, ids in CONFIGS:
        r = {'configuration': label, 'runs': len(ids), 'run_ids': ' '.join(ids), 'trainable_params_M': round(table.loc[ids[0], 'trainable_params'] / 1e6, 2),
             'mean_epochs_run': round(table.loc[ids, 'epochs_run'].mean(), 1), 'mean_minutes_per_run': round(table.loc[ids, 'train_time_s'].mean() / 60, 1)}
        for split in ['val', 'test']:
            for k in ['accuracy', 'f1_macro']:
                mu, sd = agg(ids, split, k); r[f'{split}_{k}'] = f'{mu:.4f} ± {sd:.4f}' if len(ids) > 1 else f'{mu:.4f}'
        mu, sd = agg(ids, 'test', 'roc_auc_ovr_macro'); r['test_roc_auc'] = f'{mu:.4f} ± {sd:.4f}' if len(ids) > 1 else f'{mu:.4f}'
        ext.append(r)
    pd.DataFrame(ext).to_csv(out / 'tables' / 'all_configurations_summary.csv', index=False)

    # ---- 3. final model: report, per-class metrics with intervals
    mf, yf, Pf = M[(FINAL, 'test')]
    ci = bootstrap_ci(yf, Pf.argmax(1), n_boot=2000, seed=42)
    pc = mf['per_class'].copy(); cm = mf['confusion_matrix']
    pc['recall_95CI_low'], pc['recall_95CI_high'] = zip(*[wilson(cm[i, i], cm[i].sum()) for i in range(4)])
    pc.round(4).to_csv(out / 'tables' / 'final_model_per_class_test.csv', index=False)
    (out / 'tables' / 'final_model_classification_report_test.txt').write_text(classification_report(yf, Pf.argmax(1), target_names=CL, digits=4))
    S['final_model_test'] = {**{k: round(v, 4) for k, v in mf['summary'].items() if k != 'n'}, 'n': mf['summary']['n'], 'bootstrap_ci_accuracy': ci['accuracy_ci'], 'bootstrap_ci_macroF1': ci['f1_macro_ci']}
    S['final_model_per_class_recall'] = {c: round(float(pc.set_index('cls').loc[c, 'recall_sensitivity']), 4) for c in CL}
    pcs = []
    for name, ids in [('CNN baseline (B1)', ['B1'])] + [(k, v) for k, v in FULL.items()]:
        d = pd.concat([M[(e, 'test')][0]['per_class'].set_index('cls')[['precision', 'recall_sensitivity', 'specificity', 'f1', 'roc_auc_ovr']] for e in ids], keys=ids)
        g = d.groupby(level=1, sort=False).agg(['mean', 'std']); g.columns = [f'{a}_{b}' for a, b in g.columns]; g.insert(0, 'model', name); pcs.append(g.reset_index().rename(columns={'cls': 'class'}))
    pd.concat(pcs).round(4).to_csv(out / 'tables' / 'per_class_by_model_test.csv', index=False)

    # ---- 4. confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
    heatmap(ax[0], cm, f'Final model (ResNet18, seed 42), test set, n={len(yf)}')
    heatmap(ax[1], M[('B1', 'test')][0]['confusion_matrix'], 'CNN baseline (B1), test set')
    fig.tight_layout(); savefig(fig, out / 'figures' / 'confusion_matrix_final_vs_baseline.png')
    pooled = {k: sum(M[(e, 'test')][0]['confusion_matrix'] for e in ids) for k, ids in FULL.items()}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for a_, (k, c_) in zip(ax, pooled.items()): heatmap(a_, c_, f'{k}, 3 seeds pooled (counts are sums over 3 runs)')
    fig.tight_layout(); savefig(fig, out / 'figures' / 'confusion_matrix_pooled_3_seeds.png')
    for k, c_ in pooled.items(): pd.DataFrame(c_, index=CL, columns=CL).to_csv(out / 'tables' / f'confusion_matrix_pooled_{k}.csv')
    pd.DataFrame(cm, index=CL, columns=CL).to_csv(out / 'tables' / 'confusion_matrix_final_model_test.csv')

    # ---- 5. ROC curves, final model
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.8)); ls = {'Glioma': '-', 'Meningioma': '--', 'No Tumor': '-.', 'Pituitary': ':'}
    for a_, zoom in zip(ax, [False, True]):
        for k, c in enumerate(CL):
            fpr, tpr, _ = roc_curve((yf == k).astype(int), Pf[:, k])
            a_.plot(fpr, tpr, color=CLASS_COL[c], ls=ls[c], lw=2, label=f'{c} (AUC {auc(fpr, tpr):.4f})')
        if not zoom: a_.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (1, 3))); a_.set_title('One-vs-rest ROC, final model, test set')
        else: a_.set_xlim(0, .1); a_.set_ylim(.8, 1.005); a_.set_title('Zoom: false positive rate 0 to 0.1')
        a_.set_xlabel('False positive rate'); a_.set_ylabel('True positive rate')
    ax[0].legend(loc='lower right', fontsize=8.5)
    fig.tight_layout(); savefig(fig, out / 'figures' / 'roc_final_model.png')

    # ---- 6. model comparison chart (each dot is one training run)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True); ypos = np.arange(len(CONFIGS))[::-1]
    for a_, split, title in zip(ax, ['test', 'val'], ['Test macro F1', 'Validation macro F1']):
        for (label, fam, ids), y0 in zip(CONFIGS, ypos):
            v = allruns.loc[ids, f'{split}_f1_macro'].values; col = FAMILY_COL[fam]
            a_.plot([v.min(), v.max()], [y0, y0], color=col, lw=2.2, alpha=.45, solid_capstyle='round')
            off = np.linspace(-.16, .16, len(v)) if len(v) > 1 else np.zeros(1)
            a_.scatter(v, y0 + off, s=46, color=col, edgecolor='white', linewidth=1.5, zorder=3)
            a_.text(v.max() + .004, y0, f'{v.mean():.3f}' + (f' (mean of {len(v)})' if len(v) > 1 else ''), va='center', fontsize=8.3, color=INK2)
        a_.set_title(title); a_.set_xlim(.86, 1.045); a_.set_xticks(np.arange(.86, 1.001, .02)); a_.set_xlabel('Macro F1')
        a_.grid(axis='y', visible=False)
    ax[0].set_yticks(ypos); ax[0].set_yticklabels([c[0] for c in CONFIGS])
    fig.suptitle('Each dot is one training run; lines span the 3 seeds where seeds were repeated', fontsize=9.5, color=INK2, y=1.0)
    fig.tight_layout(); savefig(fig, out / 'figures' / 'model_comparison_macro_f1.png')

    # ---- 7. tuning chart (Stage 6a), validation only
    ref_ids = FULL['ResNet18']; ref = allruns.loc[ref_ids, 'val_f1_macro']; thr = ref.mean() + max(ref.max() - ref.min(), .005)
    names = {'T1-lr1e-5': 'T1 pretrained-layer lr 1e-5', 'T2-lr1e-3': 'T2 pretrained-layer lr 1e-3', 'T3-bs16': 'T3 batch size 16', 'T4-adamw': 'T4 AdamW, weight decay 0.01',
             'T5-dropout': 'T5 dropout 0.3', 'T6-weighted': 'T6 class-weighted loss', 'T7-hflip': 'T7 horizontal flip'}
    fig, ax = plt.subplots(figsize=(8.6, 4.4)); labels = ['Reference (3 seeds)', 'Reference, Stage 4 run'] + [names[t] for t in TUNING]; yy = np.arange(len(labels))[::-1]
    ax.scatter(ref.values, np.full(3, yy[0]), s=46, color='#2a78d6', edgecolor='white', linewidth=1.5, zorder=3)
    ax.scatter([allruns.loc['R18-C-orig', 'val_f1_macro']], [yy[1]], s=46, facecolor='white', edgecolor='#2a78d6', linewidth=1.8, zorder=3)
    for t, y0 in zip(TUNING, yy[2:]): ax.scatter([allruns.loc[t, 'val_f1_macro']], [y0], s=46, color='#eb6834', edgecolor='white', linewidth=1.5, zorder=3)
    ax.axvline(ref.mean(), color='#2a78d6', lw=1.2, ls='--'); ax.axvline(thr, color=INK2, lw=1.2, ls=':')
    ax.text(ref.mean() - .0008, yy[-1] - .75, f'reference mean {ref.mean():.4f}', ha='right', fontsize=8, color='#2a78d6')
    ax.text(thr + .0004, yy[-1] - .75, f'adoption threshold {thr:.4f}', ha='left', fontsize=8, color=INK2)
    ax.set_yticks(yy); ax.set_yticklabels(labels); ax.set_xlim(.968, 1.0); ax.set_ylim(yy[-1] - 1.2, yy[0] + .6); ax.set_xlabel('Validation macro F1 (563 images; 1 image is about 0.002)')
    ax.set_title('Stage 6a: no single change reached the pre-set adoption threshold'); ax.grid(axis='y', visible=False)
    fig.tight_layout(); savefig(fig, out / 'figures' / 'tuning_validation_macro_f1.png')

    # ---- 8. training curves
    hist = {e: pd.read_csv(res / 'metrics' / f'{e}_history.csv') for e in ['B1', 'R18-C', 'R50-C']}
    cols_ = {'B1': FAMILY_COL['CNN from scratch'], 'R18-C': FAMILY_COL['ResNet18'], 'R50-C': FAMILY_COL['ResNet50']}
    lab_ = {'B1': 'CNN from scratch (B1)', 'R18-C': 'ResNet18 full (R18-C)', 'R50-C': 'ResNet50 full (R50-C)'}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for e, h in hist.items():
        b = int(h.val_macro_f1.idxmax())
        ax[0].plot(h.epoch, h.val_loss, color=cols_[e], lw=1.8, label=lab_[e]); ax[1].plot(h.epoch, h.val_macro_f1, color=cols_[e], lw=1.8, label=lab_[e])
        ax[0].scatter([h.epoch[b]], [h.val_loss[b]], color=cols_[e], edgecolor='white', s=55, zorder=3); ax[1].scatter([h.epoch[b]], [h.val_macro_f1[b]], color=cols_[e], edgecolor='white', s=55, zorder=3)
    ax[0].set_ylabel('Validation loss'); ax[1].set_ylabel('Validation macro F1'); ax[1].set_ylim(.4, 1.0)
    for a_ in ax: a_.set_xlabel('Epoch')
    ax[0].set_title('Validation loss (dot = epoch of the saved checkpoint)'); ax[1].set_title('Validation macro F1'); ax[1].legend(loc='lower right')
    fig.tight_layout(); savefig(fig, out / 'figures' / 'training_curves_comparison.png')
    h = hist['R18-C']; fb = int(h.val_macro_f1.idxmax())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(h.epoch, h.train_loss, color='#2a78d6', lw=1.8, label='Training loss'); ax[0].plot(h.epoch, h.val_loss, color='#eb6834', lw=1.8, label='Validation loss')
    ax[1].plot(h.epoch, h.train_acc, color='#2a78d6', lw=1.8, label='Training accuracy (augmented batches)'); ax[1].plot(h.epoch, h.val_acc, color='#eb6834', lw=1.8, label='Validation accuracy')
    for a_, t_ in zip(ax, ['Loss', 'Accuracy']):
        a_.axvline(h.epoch[fb], color=INK2, ls=':', lw=1.2, label=f'Saved checkpoint (epoch {int(h.epoch[fb])})'); a_.set_xlabel('Epoch'); a_.set_title(f'Final model (R18-C): {t_}'); a_.legend()
    ax[0].set_ylabel('Cross-entropy loss'); ax[1].set_ylabel('Accuracy')
    fig.tight_layout(); savefig(fig, out / 'figures' / 'training_curves_final_model.png')

    # ---- 9. error analysis
    idx = load_pred(res, FINAL, 'test')[0].index
    meta = splits.loc[idx].copy()
    runs6 = FULL['ResNet18'] + FULL['ResNet50']
    wrong = pd.DataFrame({e: ~load_pred(res, e, 'test')[0].loc[idx, 'correct'].values for e in runs6 + ['B1']}, index=idx)
    meta['n_wrong_6'] = wrong[runs6].sum(1).values; meta['wrong_final'] = wrong[FINAL].values; meta['wrong_B1'] = wrong['B1'].values
    meta['group_size'] = meta.groupby('group').group.transform('size') if 'group' in meta else 1
    gsz = splits.groupby('group').size(); meta['group_size'] = meta.group.map(gsz).values
    meta['tumour_box_area'] = [max([b[3] * b[4] for b in json.loads(j)], default=np.nan) if l != 'No Tumor' else np.nan for j, l in zip(meta.boxes_json, meta.label)]
    meta['size_group'] = np.where((meta.width == 512) & (meta.height == 512), '512x512', 'other size')
    meta['source_style'] = meta.stem.map(source_style)
    pf = load_pred(res, FINAL, 'test')[0].loc[idx]
    meta['pred_final'] = pf.pred_label.values; meta['p_pred_final'] = pf[[f'prob_{c}' for c in CL]].max(axis=1).values
    meta['p_true_final'] = [pf.loc[i, f'prob_{t}'] for i, t in zip(idx, meta.label)]
    err = meta[meta.wrong_final].copy()
    keep = ['label', 'pred_final', 'p_pred_final', 'p_true_final', 'n_wrong_6', 'wrong_B1', 'width', 'height', 'std', 'mean', 'tumour_box_area', 'group_size', 'source_style']
    err[keep].rename(columns={'label': 'true_label', 'pred_final': 'predicted', 'std': 'contrast_std', 'mean': 'mean_intensity'}).round(3).to_csv(out / 'tables' / 'misclassified_final_model_test.csv')
    S['final_model_test_errors'] = int(meta.wrong_final.sum())
    S['final_errors_by_pair'] = {f'{t} -> {p}': int(n) for (t, p), n in err.groupby(['label', 'pred_final']).size().items()}
    S['final_errors_also_wrong_in_B1'] = int(err.wrong_B1.sum())
    S['images_wrong_in_all_6_runs'] = int((meta.n_wrong_6 == 6).sum()); S['images_wrong_in_at_least_1_of_6_runs'] = int((meta.n_wrong_6 >= 1).sum())
    S['images_wrong_in_5_or_6_runs'] = int((meta.n_wrong_6 >= 5).sum())

    # pooled confusions over the six full fine-tuning runs
    pool6 = sum(M[(e, 'test')][0]['confusion_matrix'] for e in runs6)
    conf = [(CL[i], CL[j], int(pool6[i, j]), pool6[i, j] / 6) for i in range(4) for j in range(4) if i != j and pool6[i, j] > 0]
    conf = pd.DataFrame(conf, columns=['true', 'predicted', 'count_over_6_runs', 'mean_per_run']).sort_values('count_over_6_runs', ascending=False)
    conf.round(2).to_csv(out / 'tables' / 'top_confusions_6_runs.csv', index=False)
    S['top_confusions_6_runs'] = conf.head(6).round(2).to_dict('records')

    # error rate by factor (pooled over the six runs; descriptive)
    fr = []
    def add(factor, level, mask):
        sub = meta[mask]
        if len(sub): fr.append(dict(factor=factor, level=level, images=len(sub), mean_error_rate_6_runs=round(sub.n_wrong_6.sum() / (6 * len(sub)), 4), images_wrong_in_any_run=int((sub.n_wrong_6 > 0).sum()), images_wrong_in_final_model=int(sub.wrong_final.sum())))
    for c in CL: add('true class', c, meta.label == c)
    for g in ['512x512', 'other size']: add('image size', g, meta.size_group == g)
    for s_ in sorted(meta.source_style.unique()): add('filename pattern (proxy for source)', s_, meta.source_style == s_)
    tum = meta.label != 'No Tumor'; qs = meta.loc[tum, 'tumour_box_area'].quantile([1 / 3, 2 / 3]).values
    add(f'box area, tumour classes (edges {qs[0]:.3f}, {qs[1]:.3f} of image)', 'smallest third', tum & (meta.tumour_box_area <= qs[0]))
    add(f'box area, tumour classes (edges {qs[0]:.3f}, {qs[1]:.3f} of image)', 'middle third', tum & (meta.tumour_box_area > qs[0]) & (meta.tumour_box_area <= qs[1]))
    add(f'box area, tumour classes (edges {qs[0]:.3f}, {qs[1]:.3f} of image)', 'largest third', tum & (meta.tumour_box_area > qs[1]))
    cq = meta['std'].quantile([1 / 3, 2 / 3]).values
    add('contrast (pixel std) tercile', 'lowest third', meta['std'] <= cq[0]); add('contrast (pixel std) tercile', 'middle third', (meta['std'] > cq[0]) & (meta['std'] <= cq[1])); add('contrast (pixel std) tercile', 'highest third', meta['std'] > cq[1])
    add('near-duplicate group', 'singleton image', meta.group_size == 1); add('near-duplicate group', 'member of a multi-image group', meta.group_size > 1)
    fr = pd.DataFrame(fr); fr.to_csv(out / 'tables' / 'error_rate_by_factor.csv', index=False)
    def fisher(mask_a, mask_b, name):
        A, B = meta[mask_a], meta[mask_b]; t = [[int((A.n_wrong_6 > 0).sum()), int((A.n_wrong_6 == 0).sum())], [int((B.n_wrong_6 > 0).sum()), int((B.n_wrong_6 == 0).sum())]]
        return dict(comparison=name, group_a_wrong_any_vs_never=t[0], group_b_wrong_any_vs_never=t[1], fisher_exact_p=round(float(fisher_exact(t)[1]), 4))
    tests = [fisher(meta.size_group == '512x512', meta.size_group == 'other size', '512x512 vs other size (image wrong in any of 6 runs)'),
             fisher(tum & (meta.tumour_box_area <= qs[0]), tum & (meta.tumour_box_area > qs[1]), 'tumour box area: smallest vs largest third'),
             fisher(meta.group_size == 1, meta.group_size > 1, 'singleton vs multi-image group')]
    pd.DataFrame(tests).to_csv(out / 'tables' / 'error_factor_tests_exploratory.csv', index=False); S['exploratory_tests'] = tests

    # confidence of correct vs wrong predictions, pooled over six runs
    cc, cw = [], []
    for e in runs6:
        p, y, P = load_pred(res, e, 'test'); top = P.max(1); ok = P.argmax(1) == y; cc += list(top[ok]); cw += list(top[~ok])
    S['confidence_6_runs'] = dict(mean_top_prob_correct=round(float(np.mean(cc)), 4), mean_top_prob_wrong=round(float(np.mean(cw)), 4), wrong_predictions=len(cw), wrong_with_top_prob_above_0_9=int(np.sum(np.array(cw) > .9)))

    # example figures
    def grid(df, path, title, ncol=4, box=True, caption=None):
        n = len(df); nrow = int(np.ceil(n / ncol)); fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.25 * nrow), squeeze=False)
        for a_ in axes.ravel(): a_.axis('off')
        for a_, (rel, r) in zip(axes.ravel(), df.iterrows()):
            if r.wrong_final: ttl = f'{r.label} → {r.pred_final}\np = {r.p_pred_final:.2f} | wrong in {int(r.n_wrong_6)}/6 runs'
            elif r.n_wrong_6 > 0: ttl = f'{r.label}: final model correct\np = {r.p_pred_final:.2f} | wrong in {int(r.n_wrong_6)}/6 runs'
            else: ttl = f'{r.label} (correct)\np = {r.p_pred_final:.2f}'
            draw_img(a_, root, rel, r, box and r.label != 'No Tumor', ttl)
        fig.suptitle(title, fontsize=10.5, y=1.0 + .02 * (3 / max(nrow, 1)))
        if caption: fig.text(.5, -.01, caption, ha='center', fontsize=8, color=INK2)
        fig.tight_layout(h_pad=2.6, w_pad=1.0); savefig(fig, path)
    cap = 'Dashed yellow boxes are the dataset annotations (tumour classes only). They are shown for context and were not used for training.'
    grid(err.sort_values(['label', 'pred_final']), out / 'figures' / 'misclassified_examples_final_model.png', f'All {len(err)} test images misclassified by the final model', caption=cap)
    hard = meta[meta.n_wrong_6 >= 5].sort_values(['label', 'n_wrong_6'])
    if len(hard): grid(hard.head(12), out / 'figures' / 'hard_examples_wrong_in_5_or_6_runs.png', f'Test images misclassified in at least 5 of the 6 ResNet runs (final-model prediction shown)', caption=cap)
    rng = np.random.default_rng(42); ok = meta[~meta.wrong_final]; pick = []
    for c in CL: pick.append(ok[ok.label == c].sample(1, random_state=int(rng.integers(0, 10**6))))
    grid(pd.concat(pick), out / 'figures' / 'correct_examples_final_model.png', 'Correctly classified test images (one random image per class, not hand-picked)', caption=cap)

    # ---- 10. figure: error rate by factor
    sub = fr[fr.factor.isin(['true class', 'image size', 'near-duplicate group'])]
    fig, ax = plt.subplots(figsize=(7.4, 3.4)); yy = np.arange(len(sub))[::-1]
    ax.barh(yy, sub.mean_error_rate_6_runs * 100, color='#2a78d6', height=.62)
    for y0, (_, r) in zip(yy, sub.iterrows()): ax.text(r.mean_error_rate_6_runs * 100 + .08, y0, f'{r.mean_error_rate_6_runs * 100:.2f}%  (n={r.images})', va='center', fontsize=8.3, color=INK2)
    ax.set_yticks(yy); ax.set_yticklabels([f'{f.split(" (")[0]}: {l}' for f, l in zip(sub.factor, sub.level)]); ax.set_xlabel('Mean test error rate over the 6 ResNet runs (%)'); ax.grid(axis='y', visible=False); ax.set_xlim(0, sub.mean_error_rate_6_runs.max() * 100 * 1.5)
    ax.set_title('Where the ResNet errors fall (descriptive; small counts)')
    fig.tight_layout(); savefig(fig, out / 'figures' / 'error_rate_by_factor.png')

    # ---- 11. cost
    cost = pd.DataFrame([{'configuration': l, 'trainable_params_M': round(table.loc[i[0], 'trainable_params'] / 1e6, 2), 'mean_epochs': round(table.loc[i, 'epochs_run'].mean(), 1),
                          'mean_minutes_per_run': round(table.loc[i, 'train_time_s'].mean() / 60, 1), 'mean_sec_per_epoch': round((table.loc[i, 'train_time_s'] / table.loc[i, 'epochs_run']).mean(), 1)} for l, f, i in CONFIGS])
    cost.to_csv(out / 'tables' / 'computational_cost.csv', index=False)
    json.dump(S, open(out / 'stage7_summary.json', 'w'), indent=2, default=str)
    print(json.dumps(S, indent=1, default=str)[:6000])
    print('\nwritten:', sorted(p.name for p in out.rglob('*') if p.is_file()))


if __name__ == '__main__':
    main()
