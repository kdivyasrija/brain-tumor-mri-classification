"""Shortcut diagnostic: how well can the class be predicted WITHOUT looking at anatomy?
Uses only image metadata and global pixel statistics. Fit on the train split, evaluated on the
validation split (the test split is not touched). A high score means a CNN could also exploit
such non-anatomical cues, so its accuracy must be interpreted with care."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

FEATURE_SETS = {
    'majority class (reference)': [],
    'image size only (width, height, aspect, is_512x512)': ['width', 'height', 'aspect', 'is_512'],
    'size + colour mode + file size': ['width', 'height', 'aspect', 'is_512', 'mode_code', 'gray_like', 'size_kb'],
    'size + mode + file size + global intensity (mean, std)': ['width', 'height', 'aspect', 'is_512', 'mode_code', 'gray_like', 'size_kb', 'mean', 'std'],
}

def run(df, seed=42):
    d = df.copy()
    d['aspect'] = d.width / d.height
    d['is_512'] = ((d.width == 512) & (d.height == 512)).astype(int)
    d['mode_code'] = d['mode'].astype('category').cat.codes
    d['gray_like'] = d.gray_like.astype(int)
    tr, va = d[d.split == 'train'], d[d.split == 'val']
    rows = []
    for name, feats in FEATURE_SETS.items():
        if feats:
            clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight='balanced')
            clf.fit(tr[feats], tr.label_idx); pred = clf.predict(va[feats])
        else:
            clf = DummyClassifier(strategy='most_frequent').fit(tr[['width']], tr.label_idx); pred = clf.predict(va[['width']])
        rows.append(dict(features=name, val_accuracy=accuracy_score(va.label_idx, pred),
                         val_macro_f1=f1_score(va.label_idx, pred, average='macro')))
        last_pred = pred
    return pd.DataFrame(rows), last_pred, va
