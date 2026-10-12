"""PyTorch Dataset and DataLoader construction."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from preprocessing import get_transforms


class MRIDataset(Dataset):
    """Reads rows of the split table. Images are opened lazily and converted to RGB."""
    def __init__(self, df, root, transform=None, return_meta=False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row.rel).convert('RGB')   # L / RGBA / P -> RGB
        if self.transform is not None:
            img = self.transform(img)
        label = int(row.label_idx)
        if self.return_meta:
            return img, label, dict(rel=row.rel, boxes=json.loads(row.boxes_json))
        return img, label


def _worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); import random; random.seed(seed)


def class_weights(train_df, n_classes=4):
    """Inverse-frequency weights (only used if the imbalance experiment is run)."""
    counts = np.bincount(train_df.label_idx.values, minlength=n_classes).astype(float)
    return torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32)


def make_loaders(split_df, root, batch_size=32, img_size=224, hflip=False, num_workers=2, seed=42, sampler=None):
    """Return dict of train/val/test loaders. Only the training set uses random augmentation."""
    g = torch.Generator(); g.manual_seed(seed)
    loaders = {}
    for name in ['train', 'val', 'test']:
        part = split_df[split_df.split == name]
        ds = MRIDataset(part, root, get_transforms(train=(name == 'train'), img_size=img_size, hflip=hflip))
        loaders[name] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(name == 'train' and sampler is None),
            sampler=sampler if name == 'train' else None,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init, generator=g if name == 'train' else None,
            drop_last=False)
    return loaders
