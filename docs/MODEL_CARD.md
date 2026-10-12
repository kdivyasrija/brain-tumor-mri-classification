# Model card: brain MRI four-class classifier (student research project)

**Not a medical device. Not validated for clinical use. Do not use it to make or support diagnostic or treatment decisions.**

## Model
- ResNet18 (torchvision), ImageNet-pretrained, all layers fine-tuned, final layer replaced by a 4-way linear layer.
- Classes (output order): Glioma, Meningioma, No Tumor, Pituitary.
- Input: a 2D MRI slice (JPEG/PNG). Preprocessing: pad to a centred black square, resize to 224 x 224, scale to [0,1], ImageNet mean/std normalisation. No random operations at inference.
- Training: Adam, learning rate 1e-3 for the new layer and 1e-4 for pretrained layers, batch size 32, ReduceLROnPlateau, early stopping on validation macro-F1, conservative augmentation (small crop, rotation of at most 10 degrees, 5% shift, slight brightness/contrast). No horizontal flip.
- Checkpoint: run R18-C, epoch 14, selected on validation macro-F1 (0.9932).

## Data
Kaggle dataset `ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes`. Exact duplicates were removed, near-duplicates were grouped, and a stratified group-aware split of about 70/15/15 was used. The dataset has no patient identifiers.

## Measured performance (held-out test split, n = 544, one checkpoint)
Accuracy 0.9798, macro-F1 0.9797, one-vs-rest macro ROC-AUC 0.9994. Mean over three seeds of the same recipe: accuracy 0.9767, macro-F1 0.9773. Per-class results are in `final_model_selection.md`.

## Known limitations
- Possible same-patient images across splits and source heterogeneity could not be ruled out, so the accuracy may be optimistic. No external test set was used.
- Lowest sensitivity for Meningioma. Some errors have high predicted probability.
- Grad-CAM heatmaps agree only weakly with the annotated boxes and must not be read as tumour localisation.
- Only four classes, one dataset, 2D slices. Behaviour on other scanners, sequences, planes, ages, or post-operative images is unknown.
