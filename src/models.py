"""Model definitions. Stage 3: baseline CNN. (ResNet builders are added in Stage 4.)"""
import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool(2). Halves the spatial size."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class BaselineCNN(nn.Module):
    """Small CNN trained from scratch (no pretrained weights).
    5 conv blocks (32-64-128-256-256 filters) -> global average pooling -> dropout -> linear(4).
    For a 224x224 input the feature map is 7x7 before global pooling."""
    def __init__(self, n_classes=4, dropout=0.5):
        super().__init__()
        chans = [3, 32, 64, 128, 256, 256]
        self.features = nn.Sequential(*[conv_block(chans[i], chans[i + 1]) for i in range(5)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(chans[-1], n_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------------------------------------------
# Stage 4/5: ImageNet-pretrained ResNets (torchvision)
# ----------------------------------------------------------------------------------------------
RESNETS = {'resnet18': ('resnet18', 'ResNet18_Weights'), 'resnet50': ('resnet50', 'ResNet50_Weights')}


def build_resnet(name='resnet18', n_classes=4, strategy='frozen', pretrained=True, dropout=0.0):
    """ImageNet-pretrained ResNet with a new n_classes output layer.
    strategy:
      'frozen' : every pretrained layer frozen; only the new fc layer is trained (feature extractor)
      'layer4' : last residual stage (layer4) + fc trained; everything earlier frozen
      'full'   : all layers trained (full fine-tuning)
    The new fc layer is created AFTER freezing, so it is always trainable."""
    import torchvision.models as tvm
    ctor_name, weights_name = RESNETS[name]
    weights = getattr(tvm, weights_name).IMAGENET1K_V1 if pretrained else None
    model = getattr(tvm, ctor_name)(weights=weights)
    if strategy not in ('frozen', 'layer4', 'full'):
        raise ValueError(f'unknown strategy {strategy}')
    for pname, p in model.named_parameters():
        if strategy == 'frozen':
            p.requires_grad = False
        elif strategy == 'layer4':
            p.requires_grad = pname.startswith('layer4')
        else:
            p.requires_grad = True
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, n_classes)) if dropout > 0 else nn.Linear(in_f, n_classes)
    return model


def get_param_groups(model, head_lr, backbone_lr):
    """Separate learning rates: a larger one for the newly initialised fc layer, a smaller one for pretrained layers."""
    head = [p for n, p in model.named_parameters() if n.startswith('fc.') and p.requires_grad]
    backbone = [p for n, p in model.named_parameters() if not n.startswith('fc.') and p.requires_grad]
    groups = [dict(params=head, lr=head_lr, name='head')]
    if backbone:
        groups.append(dict(params=backbone, lr=backbone_lr, name='backbone'))
    return groups


def trainable_summary(model):
    """Number of trainable parameters per top-level block, for printing and for sanity assertions."""
    from collections import OrderedDict
    out = OrderedDict()
    for n, p in model.named_parameters():
        if p.requires_grad:
            out[n.split('.')[0]] = out.get(n.split('.')[0], 0) + p.numel()
    return dict(out)
