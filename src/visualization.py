"""Report-ready plots (matplotlib / seaborn only)."""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def _save(fig, path):
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches='tight')


def plot_history(hist, title, path=None, best_epoch=None):
    """Loss curves (train/val) and accuracy curves (train/val), one panel each."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.epoch, hist.train_loss, label='Training loss'); ax[0].plot(hist.epoch, hist.val_loss, label='Validation loss')
    ax[1].plot(hist.epoch, hist.train_acc, label='Training accuracy (augmented batches)'); ax[1].plot(hist.epoch, hist.val_acc, label='Validation accuracy')
    ax[1].plot(hist.epoch, hist.val_macro_f1, '--', label='Validation macro F1')
    for a, t, yl in zip(ax, ['Loss', 'Accuracy / F1'], ['Cross-entropy loss', 'Score']):
        if best_epoch: a.axvline(best_epoch, color='gray', ls=':', label=f'Best epoch ({best_epoch})')
        a.set_xlabel('Epoch'); a.set_ylabel(yl); a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); _save(fig, path); return fig


def plot_confusion_matrix(cm, class_names, title, path=None):
    """Counts (left) and row-normalised percentages = recall per class (right)."""
    cm = np.asarray(cm); norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax[0], cbar=False)
    sns.heatmap(norm * 100, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax[1], cbar=False, vmin=0, vmax=100)
    ax[0].set_title('Counts'); ax[1].set_title('Row-normalised (% of true class)')
    for a in ax: a.set_xlabel('Predicted'); a.set_ylabel('True'); a.tick_params(axis='x', rotation=20)
    fig.suptitle(title); fig.tight_layout(); _save(fig, path); return fig


def plot_roc(y_true, probs, class_names, title, path=None):
    """One-vs-rest ROC curve for every class."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for k, c in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == k).astype(int), probs[:, k])
        ax.plot(fpr, tpr, label=f'{c} (AUC = {auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.set_title(title); ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=.3)
    fig.tight_layout(); _save(fig, path); return fig


def plot_per_class(per_class, title, path=None):
    """Grouped bars: precision, recall, F1 for every class."""
    d = per_class.melt(id_vars='cls', value_vars=['precision', 'recall_sensitivity', 'f1'], var_name='metric', value_name='value')
    d['metric'] = d.metric.replace({'recall_sensitivity': 'recall'})
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.barplot(data=d, x='cls', y='value', hue='metric', ax=ax)
    for c in ax.containers: ax.bar_label(c, fmt='%.2f', fontsize=7)
    ax.set_ylim(0, 1.08); ax.set_xlabel('Class'); ax.set_ylabel('Score'); ax.set_title(title); ax.legend(title='')
    fig.tight_layout(); _save(fig, path); return fig


def plot_experiment_bars(table, path=None, title='Validation and test macro F1 by experiment'):
    """Grouped bars (validation vs test macro F1) for every row of the experiment table."""
    d = table.melt(id_vars='experiment_id', value_vars=['val_macro_f1', 'test_macro_f1'], var_name='split', value_name='macro_f1')
    d['split'] = d.split.map({'val_macro_f1': 'validation', 'test_macro_f1': 'test'})
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(table) + 2), 4))
    sns.barplot(data=d, x='experiment_id', y='macro_f1', hue='split', ax=ax)
    for c in ax.containers: ax.bar_label(c, fmt='%.3f', fontsize=7)
    lo = max(0.0, float(d.macro_f1.min()) - 0.1)
    ax.set_ylim(lo, 1.02); ax.set_xlabel('Experiment'); ax.set_ylabel('Macro F1'); ax.set_title(title); ax.legend(title='')
    fig.tight_layout(); _save(fig, path); return fig
