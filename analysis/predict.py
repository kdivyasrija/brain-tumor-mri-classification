"""Inference with the final model (ResNet18, run R18-C).  RESEARCH USE ONLY, not a medical device.

    python predict.py --ckpt R18-C_best.pt --images scan1.jpg scan2.png [--csv out.csv] [--device cpu]

Output per image: predicted class and the four class probabilities. Probabilities are not calibrated and
confident errors occur (see MODEL_CARD.md)."""
import argparse, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms as T
from PIL import Image

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class PadToSquare:
    def __call__(self, img):
        w, h = img.size; s = max(w, h)
        if w == h: return img
        canvas = Image.new(img.mode, (s, s), (0,) * len(img.getbands())); canvas.paste(img, ((s - w) // 2, (s - h) // 2)); return canvas


TF = T.Compose([PadToSquare(), T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


def load_model(ckpt, device='cpu'):
    ck = torch.load(ckpt, map_location=device, weights_only=False)   # a checkpoint produced by this project; only load files you trust
    m = tvm.resnet18(weights=None); m.fc = nn.Linear(512, 4); m.load_state_dict(ck['model_state']); return m.to(device).eval()


@torch.no_grad()
def predict(model, paths, device='cpu', batch_size=32):
    out = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([TF(Image.open(p).convert('RGB')) for p in paths[i:i + batch_size]]).to(device)
        out.append(torch.softmax(model(x), 1).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--csv'); ap.add_argument('--device', default='cpu'); a = ap.parse_args()
    print('RESEARCH PROTOTYPE: not for clinical use.', file=sys.stderr)
    P = predict(load_model(a.ckpt, a.device), a.images, a.device)
    rows = [dict(image=p, prediction=CLASS_NAMES[int(P[i].argmax())], **{f'prob_{c}': float(P[i, k]) for k, c in enumerate(CLASS_NAMES)}) for i, p in enumerate(a.images)]
    for r in rows: print(f"{r['image']}: {r['prediction']}  " + '  '.join(f"{c}={r['prob_' + c]:.3f}" for c in CLASS_NAMES))
    if a.csv:
        import csv
        with open(a.csv, 'w', newline='') as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
