"""Prediction, metrics, saving of results and the experiment table."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, f1_score)


def predict(model, loader, device, criterion=None):
    """Run the model on a loader WITHOUT shuffling. Returns labels, predictions, softmax probabilities,
    optional mean loss, and the relative file path of every image (same order)."""
    import torch
    model.eval()
    ys, logits, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            out = model(x).float()
            if criterion is not None:
                loss_sum += criterion(out, y).item() * x.size(0)
            n += x.size(0); logits.append(out.cpu()); ys.append(y.cpu())
    probs = torch.softmax(torch.cat(logits), dim=1).numpy().astype(np.float64)
    y_true = torch.cat(ys).numpy()
    rels = loader.dataset.df.rel.values
    assert len(rels) == len(y_true), 'loader order / length mismatch'
    return dict(y_true=y_true, y_pred=probs.argmax(1), probs=probs,
                loss=(loss_sum / n) if criterion is not None else None, rels=rels)


def compute_metrics(y_true, probs, class_names):
    """All metrics are computed from the labels and probabilities passed in."""
    K = len(class_names); labels = list(range(K))
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = cm.sum() - tp - fn - fp
    sens = tp / np.maximum(tp + fn, 1)      # = recall of the class
    spec = tn / np.maximum(tn + fp, 1)      # one-vs-rest specificity
    pc = pd.DataFrame(dict(cls=class_names, precision=p, recall_sensitivity=r, specificity=spec, f1=f, support=s))
    P = probs / probs.sum(1, keepdims=True)
    try:
        auc_macro = roc_auc_score(y_true, P, multi_class='ovr', average='macro', labels=labels)
        auc_weighted = roc_auc_score(y_true, P, multi_class='ovr', average='weighted', labels=labels)
        auc_pc = [roc_auc_score((y_true == k).astype(int), P[:, k]) for k in labels]
    except ValueError:
        auc_macro = auc_weighted = float('nan'); auc_pc = [float('nan')] * K
    pc['roc_auc_ovr'] = auc_pc
    summary = dict(n=int(len(y_true)), accuracy=accuracy_score(y_true, y_pred),
                   balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
                   precision_macro=mp, recall_macro=mr, f1_macro=mf,
                   precision_weighted=wp, recall_weighted=wr, f1_weighted=wf,
                   roc_auc_ovr_macro=auc_macro, roc_auc_ovr_weighted=auc_weighted)
    return dict(summary={k: float(v) if not isinstance(v, int) else v for k, v in summary.items()},
                per_class=pc, confusion_matrix=cm)


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42, alpha=0.05):
    """Percentile bootstrap CI (resampling test images) for accuracy and macro-F1.
    It reflects test-set sampling noise only, not training randomness."""
    rng = np.random.default_rng(seed); n = len(y_true); acc, f1 = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        acc.append((y_true[i] == y_pred[i]).mean()); f1.append(f1_score(y_true[i], y_pred[i], average='macro'))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return dict(accuracy_ci=tuple(np.percentile(acc, q).round(4)), f1_macro_ci=tuple(np.percentile(f1, q).round(4)))


def evaluate_split(model, loader, device, class_names, tag, out_root, criterion=None, n_boot=1000, seed=42):
    """Predict, compute metrics, and save metrics / per-class table / confusion matrix / predictions."""
    out_root = Path(out_root)
    res = predict(model, loader, device, criterion)
    m = compute_metrics(res['y_true'], res['probs'], class_names)
    m['summary']['loss'] = res['loss']
    m['bootstrap_ci'] = bootstrap_ci(res['y_true'], res['y_pred'], n_boot, seed)
    save_results(tag, res, m, class_names, out_root)
    return m, res


def save_results(tag, res, m, class_names, out_root):
    out_root = Path(out_root)
    (out_root / 'metrics').mkdir(parents=True, exist_ok=True); (out_root / 'predictions').mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=m['summary'], bootstrap_ci=m.get('bootstrap_ci')), open(out_root / 'metrics' / f'{tag}_metrics.json', 'w'), indent=2, default=str)
    m['per_class'].to_csv(out_root / 'metrics' / f'{tag}_per_class.csv', index=False)
    pd.DataFrame(m['confusion_matrix'], index=class_names, columns=class_names).to_csv(out_root / 'metrics' / f'{tag}_confusion_matrix.csv')
    pred = pd.DataFrame(dict(rel=res['rel'] if 'rel' in res else res['rels'], y_true=res['y_true'], y_pred=res['y_pred']))
    pred['true_label'] = [class_names[i] for i in pred.y_true]; pred['pred_label'] = [class_names[i] for i in pred.y_pred]
    pred['correct'] = pred.y_true == pred.y_pred
    for k, c in enumerate(class_names):
        pred[f'prob_{c}'] = res['probs'][:, k]
    pred.to_csv(out_root / 'predictions' / f'{tag}_predictions.csv', index=False)


EXPERIMENT_COLUMNS = ['experiment_id', 'model', 'learning_rate', 'batch_size', 'optimizer', 'weight_decay', 'loss',
                      'max_epochs', 'best_epoch', 'epochs_run', 'finetune_strategy', 'augmentation',
                      'trainable_params', 'val_accuracy', 'val_macro_f1', 'test_accuracy', 'test_macro_f1',
                      'roc_auc_test_ovr_macro', 'train_time_s', 'seed']


def log_experiment(path, row):
    """Add or replace one row (by experiment_id) in the experiment table. Only measured values go in."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=EXPERIMENT_COLUMNS)
    df = df[df.experiment_id != row['experiment_id']]
    df = pd.concat([df, pd.DataFrame([row]).reindex(columns=EXPERIMENT_COLUMNS)], ignore_index=True)
    df.to_csv(path, index=False)
    return df
