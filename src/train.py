"""Training loop: early stopping, best-model checkpointing, LR scheduling."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from evaluate import predict


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    """One pass over the training loader. The returned accuracy is a running average over AUGMENTED
    batches while the weights are changing, so it is not the same as accuracy on clean training images."""
    model.train()
    # BatchNorm layers whose parameters are frozen stay in eval mode, so the pretrained running
    # statistics are not overwritten. (No effect when every layer is trainable.)
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x); loss = criterion(out, y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        loss_sum += loss.item() * x.size(0); correct += (out.argmax(1) == y).sum().item(); n += x.size(0)
    return loss_sum / n, correct / n


def fit(model, loaders, criterion, optimizer, device, epochs=40, patience=8, scheduler=None,
        ckpt_path='best.pt', monitor='val_macro_f1', use_amp=True, config=None, val_criterion=None):
    """Train with validation after every epoch.
    * The checkpoint is saved whenever the monitored VALIDATION metric improves (never training accuracy).
    * Training stops after `patience` epochs without improvement.
    * ReduceLROnPlateau schedulers are stepped with the validation loss.
    * At the end the best weights are loaded back into `model`.
    `config` must contain only plain Python values (it is stored inside the checkpoint)."""
    use_amp = bool(use_amp and device.type == 'cuda')
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    val_criterion = val_criterion or criterion
    ckpt_path = Path(ckpt_path); ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    hist, best, best_epoch, wait, t_start = [], -np.inf, 0, 0, time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device, scaler, use_amp)
        va = predict(model, loaders['val'], device, val_criterion)
        va_acc = float((va['y_true'] == va['y_pred']).mean())
        va_f1 = float(f1_score(va['y_true'], va['y_pred'], average='macro'))
        lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(va['loss'])
        row = dict(epoch=epoch, lr=lr, train_loss=tr_loss, train_acc=tr_acc, val_loss=va['loss'],
                   val_acc=va_acc, val_macro_f1=va_f1, seconds=time.time() - t0)
        hist.append(row)
        score = -row[monitor] if monitor.endswith('loss') else row[monitor]
        if score > best + 1e-4:
            best, best_epoch, wait = score, epoch, 0
            torch.save(dict(model_state=model.state_dict(), epoch=epoch, monitor=monitor,
                            monitor_value=float(row[monitor]), config=config or {}), ckpt_path)
            flag = ' *saved*'
        else:
            wait += 1; flag = ''
        print(f"Epoch {epoch:3d} | lr {lr:.2e} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va['loss']:.4f} acc {va_acc:.4f} macroF1 {va_f1:.4f} | {row['seconds']:.0f}s{flag}")
        if wait >= patience:
            print(f'Early stopping: no improvement in {monitor} for {patience} epochs.')
            break
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)   # our own file
    model.load_state_dict(ckpt['model_state'])
    summary = dict(best_epoch=best_epoch, best_monitor_value=float(ckpt['monitor_value']), monitor=monitor,
                   epochs_run=len(hist), stopped_early=len(hist) < epochs, train_time_s=time.time() - t_start,
                   amp=use_amp)
    return pd.DataFrame(hist), summary
