"""Nearest-training-image similarity for every validation/test image (leakage / heterogeneity diagnostic).
Similarity = Pearson correlation of 64x64 grayscale thumbnails. Uses only the images and the locked split."""
import sys, json, re, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy.stats import mannwhitneyu, fisher_exact
res, root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
CL = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
sp = pd.read_csv(res / 'metrics' / 'splits.csv').set_index('rel')
def thumb(rel):
    a = np.asarray(Image.open(root / rel).convert('L').resize((64, 64), Image.BILINEAR), dtype=np.float32).ravel()
    a = a - a.mean(); n = np.linalg.norm(a); return a / n if n > 0 else a
X = np.stack([thumb(r) for r in sp.index]); sp['i'] = np.arange(len(sp))
tr = sp[sp.split == 'train']; Xtr = X[tr.i.values]; ytr = np.asarray(tr.label.values, dtype=object)
def style(stem):
    if re.match(r'^(Tr|Te)-', stem): return 'Tr-/Te- prefixed'
    return 'other filename pattern'
rows = []
for split in ['val', 'test']:
    d = sp[sp.split == split]; S = X[d.i.values] @ Xtr.T
    nn = S.argmax(1); mx = S.max(1)
    # same-class best similarity and best other-class similarity
    yd = np.asarray(d.label.values, dtype=object); same = np.where(ytr[None, :] == yd[:, None], S, -1).max(1); other = np.where(ytr[None, :] != yd[:, None], S, -1).max(1)
    r = pd.DataFrame({'rel': d.index, 'split': split, 'label': d.label.values, 'max_corr_train': mx, 'nn_label': ytr[nn], 'nn_correct': ytr[nn] == yd, 'best_same_class': same, 'best_other_class': other, 'style': d.stem.map(style).values})
    rows.append(r)
R = pd.concat(rows); R.to_csv(out / 'tables' / 'nearest_train_neighbour_similarity.csv', index=False)
T = R[R.split == 'test'].set_index('rel')
print('1-nearest-neighbour classification by thumbnail correlation (train as reference):')
for s in ['val', 'test']: print(f'  {s}: accuracy {R[R.split == s].nn_correct.mean():.4f} (n={int((R.split == s).sum())})')
print('\ntest images, nearest-train similarity (Pearson r of 64x64 thumbnails):')
print(T.groupby('style').max_corr_train.describe()[['count', 'mean', '25%', '50%', '75%', 'max']].round(3).to_string())
a, b = T[T['style'] == 'Tr-/Te- prefixed'].max_corr_train, T[T['style'] != 'Tr-/Te- prefixed'].max_corr_train
print(f'  Mann-Whitney p (Tr-/Te- vs other, nearest-train correlation): {mannwhitneyu(a, b).pvalue:.2e}')
print('\n1-NN accuracy on test by filename family:', T.groupby('style').nn_correct.mean().round(3).to_dict())
# do ResNet errors relate to similarity? use images wrong in any of the 6 runs
runs6 = ['R18-C', 'R18-C-s43', 'R18-C-s44', 'R50-C', 'F-R50-s43', 'F-R50-s44']
w = sum((~pd.read_csv(res / 'predictions' / f'{e}_test_predictions.csv').set_index('rel').loc[T.index, 'correct']).astype(int) for e in runs6)
T['n_wrong_6'] = w.values
print('\nmedian nearest-train similarity: images never wrong %.3f | wrong in >=1 of 6 runs %.3f | wrong in >=4 runs %.3f' % (T[T.n_wrong_6 == 0].max_corr_train.median(), T[T.n_wrong_6 >= 1].max_corr_train.median(), T[T.n_wrong_6 >= 4].max_corr_train.median()))
print('  Mann-Whitney p (never wrong vs wrong in >=1):', f'{mannwhitneyu(T[T.n_wrong_6 == 0].max_corr_train, T[T.n_wrong_6 >= 1].max_corr_train).pvalue:.2e}')
T['sim_bin'] = pd.cut(T.max_corr_train, [0, .8, .9, .95, 1.0001], labels=['<0.80', '0.80-0.90', '0.90-0.95', '>=0.95'])
g = T.groupby('sim_bin', observed=True).agg(images=('n_wrong_6', 'size'), mean_err_rate_6_runs=('n_wrong_6', lambda x: x.sum() / (6 * len(x))), wrong_in_any_run=('n_wrong_6', lambda x: int((x > 0).sum())))
print('\nResNet error rate by similarity to the closest training image:'); print(g.round(4).to_string()); g.to_csv(out / 'tables' / 'error_rate_by_nearest_train_similarity.csv')
print('\nshare of test images with a training image at r >= 0.90:', round(float((T.max_corr_train >= .9).mean()), 3), '| by family:', T.groupby('style').apply(lambda d: round(float((d.max_corr_train >= .9).mean()), 3)).to_dict())
# within the family comparison controlling for similarity: errors among images with similarity < 0.90
lo = T[T.max_corr_train < .9]; print('\nimages with nearest-train r < 0.90 only: error rate by family:', lo.groupby('style').apply(lambda d: round(d.n_wrong_6.sum() / (6 * len(d)), 4)).to_dict(), '| n:', lo.groupby('style').size().to_dict())

# ---- figure: error rate vs similarity to nearest training image, split by filename family
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
C = {'Tr-/Te- prefixed': '#2a78d6', 'other filename pattern': '#eb6834'}
T['hi'] = np.where(T.max_corr_train >= .9, 'r >= 0.90', 'r < 0.90')
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
g = T.groupby('sim_bin', observed=True).apply(lambda d: pd.Series({'rate': 100 * d.n_wrong_6.sum() / (6 * len(d)), 'n': len(d)}))
ax[0].bar(range(len(g)), g.rate, color='#2a78d6', width=.6)
for i, (r, n) in enumerate(zip(g.rate, g.n)): ax[0].text(i, r + .1, f'{r:.1f}%\n(n={int(n)})', ha='center', va='bottom', fontsize=10)
ax[0].set_xticks(range(len(g))); ax[0].set_xticklabels(g.index); ax[0].set_ylim(0, g.rate.max() * 1.35)
ax[0].set_xlabel('Highest correlation with any training image'); ax[0].set_ylabel('Error rate, 6 fine-tuned runs pooled (%)'); ax[0].set_title('All test images', loc='left')
x = np.arange(2); w = .36
for k, (fam, col) in enumerate(C.items()):
    vals = []; ns = []
    for h in ['r < 0.90', 'r >= 0.90']:
        d = T[(T['style'] == fam) & (T.hi == h)]; vals.append(100 * d.n_wrong_6.sum() / (6 * max(len(d), 1))); ns.append(len(d))
    ax[1].bar(x + (k - .5) * w, vals, w, color=col, label=fam)
    for xi, v, n in zip(x + (k - .5) * w, vals, ns): ax[1].text(xi, v + .1, f'{v:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=9)
ax[1].set_xticks(x); ax[1].set_xticklabels(['nearest train r < 0.90', 'nearest train r >= 0.90']); ax[1].set_ylim(0, 12.5)
ax[1].set_title('By filename family', loc='left'); ax[1].legend(loc='upper right')
fig.tight_layout(); (out / 'figures').mkdir(exist_ok=True); fig.savefig(out / 'figures' / 'error_rate_by_nearest_train_similarity.png', dpi=200, bbox_inches='tight')
fam = T.groupby(['style', 'hi']).apply(lambda d: pd.Series({'images': len(d), 'error_rate_6_runs': d.n_wrong_6.sum() / (6 * len(d))})).round(4)
fam.to_csv(out / 'tables' / 'error_rate_by_similarity_and_filename_family.csv'); print('\n', fam.to_string())
