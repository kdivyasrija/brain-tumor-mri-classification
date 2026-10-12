"""Grad-CAM for the ResNet18 classifier (PyTorch). Written for Colab; keep the class API stable, the analysis script uses it.

Grad-CAM (Selvaraju et al., 2017): for a target class c and a convolutional layer with activations A_k,
    alpha_k = mean over positions of d(logit_c) / dA_k ;   CAM = ReLU( sum_k alpha_k * A_k )
For ResNet18 the layer4 output goes straight into global average pooling and one linear layer, so
d(logit_c)/dA_k(x,y) = W[c,k] / (H*W) exactly. `analytic_check` verifies the hooked gradients against this identity,
which is an independent test that the hooks capture the right tensor.
"""
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as tvm
from torchvision import transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


class PadToSquare:
    """Same operation as Stage 2: centred black square canvas."""
    def __call__(self, img):
        w, h = img.size; s = max(w, h)
        if w == h: return img
        canvas = Image.new(img.mode, (s, s), (0,) * len(img.getbands())); canvas.paste(img, ((s - w) // 2, (s - h) // 2)); return canvas


EVAL_TF = T.Compose([PadToSquare(), T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


def load_batch(paths):
    return torch.stack([EVAL_TF(Image.open(p).convert('RGB')) for p in paths])


class Explainer:
    def __init__(self, model, device):
        self.model, self.device = model.to(device).eval(), device

    @classmethod
    def from_checkpoint(cls, path, device):
        ck = torch.load(path, map_location=device, weights_only=False)   # our own file from Stage 4/6
        cfg = ck.get('config', {}) or {}
        assert cfg.get('arch', 'resnet18') == 'resnet18' and float(cfg.get('dropout', 0.0)) == 0.0, f'unexpected config {cfg}'
        m = tvm.resnet18(weights=None); m.fc = nn.Linear(512, 4)
        m.load_state_dict(ck['model_state']); return cls(m, device), ck

    def _layer(self, name): return dict(layer4=self.model.layer4, layer3=self.model.layer3)[name]

    def forward_cam(self, paths, layer='layer4', targets=None, batch_size=32):
        """Returns dict(logits, probs, preds, cams(raw, ReLU'd, upsampled to 224 x 224), acts_shape).
        targets=None -> predicted class; otherwise an int array with one target class per image."""
        store = {}
        def fwd_hook(_, __, out):
            store['a'] = out
            out.register_hook(lambda g: store.__setitem__('g', g))
        h = self._layer(layer).register_forward_hook(fwd_hook)
        L, C = [], []
        try:
            for i in range(0, len(paths), batch_size):
                x = load_batch(paths[i:i + batch_size]).to(self.device)
                with torch.enable_grad():
                    logits = self.model(x)
                    t = logits.argmax(1) if targets is None else torch.as_tensor(np.asarray(targets[i:i + batch_size]), device=self.device)
                    self.model.zero_grad(set_to_none=True)
                    logits.gather(1, t[:, None]).sum().backward()      # samples are independent (BatchNorm in eval mode)
                a, g = store['a'].detach(), store['g'].detach()
                alpha = g.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((alpha * a).sum(1, keepdim=True))
                cam = nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)[:, 0]
                L.append(logits.detach().cpu()); C.append(cam.cpu())
        finally:
            h.remove()
        logits = torch.cat(L); probs = torch.softmax(logits, 1)
        return dict(logits=logits.numpy(), probs=probs.numpy().astype(np.float64), preds=logits.argmax(1).numpy(),
                    cams=torch.cat(C).numpy().astype(np.float32), acts_shape=tuple(a.shape[1:]))

    def analytic_check(self, paths):
        """Layer4: hooked alpha must equal fc.weight[c] / (H*W); also GAP identity  mean_xy(sum_k W_ck A_k) + b_c = logit_c."""
        store = {}
        def fwd_hook(_, __, out):
            store['a'] = out; out.register_hook(lambda g: store.__setitem__('g', g))
        h = self.model.layer4.register_forward_hook(fwd_hook)
        try:
            x = load_batch(paths).to(self.device)
            with torch.enable_grad():
                logits = self.model(x); t = logits.argmax(1)
                self.model.zero_grad(set_to_none=True); logits.gather(1, t[:, None]).sum().backward()
            a, g = store['a'].detach(), store['g'].detach(); n, k, hh, ww = a.shape
            alpha = g.mean(dim=(2, 3)); Wc = self.model.fc.weight.detach()[t]; b = self.model.fc.bias.detach()[t]
            err_alpha = float((alpha - Wc / (hh * ww)).abs().max()); scale = float((Wc / (hh * ww)).abs().max())
            recon = (Wc[:, :, None, None] * a).sum(1).mean(dim=(1, 2)) + b
            err_logit = float((recon - logits.detach()[torch.arange(n), t]).abs().max())
        finally:
            h.remove()
        return dict(max_abs_alpha_error=err_alpha, alpha_scale=scale, max_abs_logit_reconstruction_error=err_logit, n=int(n))

    def randomised(self, mode, seed=0):
        """Copy with re-initialised weights. mode: 'fc' | 'layer4+fc' | 'all'. Used for the sanity check of Adebayo et al. (2018)."""
        torch.manual_seed(seed)
        m = copy.deepcopy(self.model).cpu()
        parts = {'fc': [m.fc], 'layer4+fc': [m.layer4, m.fc], 'all': [m]}[mode]
        for p in parts:
            for mod in p.modules():
                if hasattr(mod, 'reset_parameters'): mod.reset_parameters()
        return Explainer(m, self.device)
