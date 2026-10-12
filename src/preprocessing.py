"""Image preprocessing and augmentation.

Training preprocessing : pad to square -> resize -> mild random augmentation -> tensor -> normalise
Validation/test        : pad to square -> resize -> tensor -> normalise      (NO random operations)

Why pad instead of stretch: the images are not all square (for example 201x251), and stretching would
distort the shape of the head and of the tumour. Padding with black keeps the aspect ratio.
Why 3 channels: ImageNet-pretrained ResNets expect RGB input, so grayscale MRI is replicated to 3 channels
(done when the file is loaded, with PIL's convert('RGB')).
Why ImageNet mean/std: the pretrained weights were trained with this normalisation.
"""
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    """Pad a PIL image with a constant colour (default black) to a centred square."""
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        if w == h:
            return img
        canvas = Image.new(img.mode, (s, s), (self.fill,) * len(img.getbands()))
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas


def get_transforms(train, img_size=224, hflip=False):
    """Return a torchvision transform. hflip is OFF by default (anatomical orientation, see report)."""
    from torchvision import transforms as T
    base = [PadToSquare(0), T.Resize((img_size, img_size))]
    tail = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose(base + tail)
    aug = [
        T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),   # moderate crop
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),                      # small rotation / shift
        T.ColorJitter(brightness=0.1, contrast=0.1),                             # slight intensity change
    ]
    if hflip:
        aug.insert(0, T.RandomHorizontalFlip(0.5))   # only for the separate ablation experiment
    return T.Compose(base + aug + tail)


def denormalize(t):
    """Tensor (3,H,W) normalised -> numpy (H,W,3) in [0,1] for plotting."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1); std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
