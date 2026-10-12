"""Dataset inventory, duplicate handling, grouping and splitting.

This module does not need PyTorch. All numbers it reports are computed from
the files on disk. The folder layout it expects was verified in Stage 1:
    <root>/{Train,Val}/<ClassName>/{images,labels}/<file>
and it raises a clear error if the layout is different.
"""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']   # alphabetical, same order as the box class ids
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def find_dataset_root(base):
    """Return the folder that directly contains the Train and Val folders."""
    base = Path(base)
    for p in [base] + [q for q in base.rglob('*') if q.is_dir()]:
        names = {c.name.lower() for c in p.iterdir() if c.is_dir()}
        if {'train', 'val'} <= names:
            return p
    raise FileNotFoundError(f'No folder containing Train/ and Val/ found under {base}')


def _dhash(img):
    a = np.asarray(img.convert('L').resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits).view('>u8')[0])


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_boxes(txt_path):
    """YOLO lines: class cx cy w h (normalised). Returns a list of lists."""
    if txt_path is None:
        return []
    boxes = []
    for line in txt_path.read_text().strip().splitlines():
        t = line.split()
        if len(t) >= 5:
            boxes.append([int(float(t[0]))] + [float(v) for v in t[1:5]])
    return boxes


def build_inventory(root, progress=lambda x, **k: x):
    """One row per image: path, class, original split, size, hashes, boxes."""
    root = Path(root)
    label_files = {p.stem: p for p in root.rglob('*.txt')}
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT)
    rows, unreadable = [], []
    for p in progress(paths, desc='inventory'):
        rel = p.relative_to(root)
        if len(rel.parts) < 3 or rel.parts[1] not in CLASS_TO_IDX:
            raise ValueError(f'Unexpected path layout: {rel}. Expected <Split>/<Class>/images/<file>.')
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
                mode, fmt = im.mode, im.format
                dh = _dhash(im)
                rgb = np.asarray(im.convert('RGB'))
        except Exception as e:
            unreadable.append((str(rel), repr(e)))
            continue
        boxes = _read_boxes(label_files.get(p.stem))
        rows.append(dict(
            rel=str(rel), split_orig=rel.parts[0].lower(), label=rel.parts[1],
            label_idx=CLASS_TO_IDX[rel.parts[1]], stem=p.stem, width=w, height=h, mode=mode, format=fmt,
            gray_like=bool(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])),
            mean=float(rgb.mean()), std=float(rgb.std()), size_kb=p.stat().st_size / 1024,
            md5=_md5(p), dhash=dh, has_label_file=p.stem in label_files,
            n_boxes=len(boxes), boxes_json=json.dumps(boxes)))
    df = pd.DataFrame(rows)
    df['dhash'] = df['dhash'].astype(np.uint64)
    return df, unreadable


def drop_exact_duplicates(df):
    """Keep one file per identical-bytes group (files from Train/ are preferred, then by path).
    Groups whose copies carry different class labels are removed entirely (none expected)."""
    df = df.copy()
    df['_rank'] = (df.split_orig != 'train').astype(int)
    df = df.sort_values(['_rank', 'rel']).reset_index(drop=True)
    n_labels = df.groupby('md5').label.nunique()
    conflict_md5 = set(n_labels[n_labels > 1].index)
    df = df[~df.md5.isin(conflict_md5)]
    keep = ~df.md5.duplicated(keep='first')
    kept = df[keep][['md5', 'rel']].rename(columns={'rel': 'kept_rel'})
    log = df[~keep][['rel', 'label', 'split_orig', 'md5']].merge(kept, on='md5')
    out = df[keep].drop(columns='_rank').reset_index(drop=True)
    return out, log, len(conflict_md5)


def near_duplicate_groups(root, df, dhash_thr=2, corr_thr=0.95, thumb=64):
    """Group images that are visually near-identical although their bytes differ.
    A pair is linked only if (a) dHash Hamming distance <= dhash_thr AND
    (b) Pearson correlation of 64x64 grayscale thumbnails >= corr_thr.
    Linked pairs are merged into connected components (union-find)."""
    root = Path(root)
    H = df.dhash.values.astype(np.uint64)
    n = len(H)
    pop = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    cand = []
    for i in range(0, n, 500):
        x = H[i:i + 500, None] ^ H[None, :]
        d = pop[x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)].sum(axis=2)
        ii, jj = np.where(d <= dhash_thr)
        cand += [(a + i, b, int(d[a, b])) for a, b in zip(ii, jj) if a + i < b]
    cache = {}
    def thumb_of(k):
        if k not in cache:
            im = Image.open(root / df.rel.iloc[k]).convert('L').resize((thumb, thumb))
            cache[k] = np.asarray(im, dtype=np.float32).ravel()
        return cache[k]
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    pairs = []
    for a, b, d in cand:
        x, y = thumb_of(a), thumb_of(b)
        c = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float('nan')
        linked = c >= corr_thr
        pairs.append(dict(i=a, j=b, dhash_dist=d, corr=c, linked=linked,
                          label_i=df.label.iloc[a], label_j=df.label.iloc[b]))
        if linked:
            parent[find(a)] = find(b)
    group = np.array([find(k) for k in range(n)])
    _, group = np.unique(group, return_inverse=True)
    return group, pd.DataFrame(pairs)


def make_splits(df, seed=42, n_splits=7):
    """Stratified, group-aware split. With 7 folds: 1 fold test (~14.3%), 1 fold val (~14.3%),
    5 folds train (~71.4%), i.e. approximately 70/15/15. Images in the same group never cross splits."""
    from sklearn.model_selection import StratifiedGroupKFold
    df = df.copy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, te) in enumerate(sgkf.split(df, df.label_idx, df.group)):
        fold[te] = k
    df['split'] = np.where(fold == 0, 'test', np.where(fold == 1, 'val', 'train'))
    return df


def verify_splits(df, pairs=None):
    """Leakage checks on the final split table. Returns a dict of results."""
    res = {}
    res['n_images'] = len(df)
    res['counts'] = df.split.value_counts().to_dict()
    res['class_counts'] = pd.crosstab(df.split, df.label).to_dict('index')
    res['md5_shared_across_splits'] = int((df.groupby('md5').split.nunique() > 1).sum())
    res['groups_shared_across_splits'] = int((df.groupby('group').split.nunique() > 1).sum())
    res['rel_duplicated'] = int(df.rel.duplicated().sum())
    if pairs is not None and len(pairs):
        lk = pairs[pairs.linked]
        sp = df.split.values
        res['linked_near_dup_pairs'] = int(len(lk))
        res['linked_near_dup_pairs_across_splits'] = int((sp[lk.i.values] != sp[lk.j.values]).sum())
        res['linked_pairs_with_different_labels'] = int((lk.label_i != lk.label_j).sum())
    return res


def run_split_pipeline(root, seed=42, progress=lambda x, **k: x):
    """Full Stage 2 pipeline in one call: inventory -> exact de-duplication -> near-duplicate groups -> split.
    Deterministic for a given dataset, seed and library versions. Returns (splits_df, info_dict)."""
    inv, unreadable = build_inventory(root, progress)
    clean, dup_log, n_conflict = drop_exact_duplicates(inv)
    group, pairs = near_duplicate_groups(root, clean)
    clean['group'] = group
    splits = make_splits(clean, seed=seed)
    return splits, dict(n_found=len(inv), unreadable=unreadable, dup_log=dup_log, n_conflict=n_conflict, pairs=pairs)
