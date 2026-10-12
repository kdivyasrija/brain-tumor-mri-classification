# Stage 8 findings: Grad-CAM against annotated boxes (final model R18-C, test set)

## Implemented
- Grad-CAM on `layer4` (7 x 7 grid) with `layer3` (14 x 14) as a sensitivity check, target class = predicted class, ResNet18 checkpoint `R18-C` (epoch 14).
- Heatmaps mapped back to original image coordinates (undoing the black padding and the resize), min-max normalised inside the image area.
- Evaluated on test images of the three tumour classes that have at least one box of the matching class: 468 of 470 (2 had no matching box). No Tumor images were excluded because their boxes cover most of the brain.
- Baselines: a fixed centre Gaussian, heatmaps of other images of the same class (5 draws), and the expected hit rate of a uniformly random point.

## Sanity checks (all passed)
- Checkpoint reproduces the saved R18-C test predictions: label agreement 100.0%, largest probability difference 1.1e-16. The Stage 4 run (R18-C-orig) agreed on 97.2%, so the checkpoint is the Stage 6 run.
- Hooked gradients equal the exact formula for this network (largest difference 2.3e-10 on weights of size 1.6e-03).
- Heat falling on the black padding: 0.11% on average.
- Weight randomisation changes the maps (Spearman correlation with the trained map; 60 images):

| Randomised part | mean | median | all-zero maps |
|---|---|---|---|
| fc | 0.05 | -0.05 | 0 of 60 |
| layer4+fc | -0.01 | -0.00 | 0 of 60 |
| all | -0.12 | -0.25 | 17 of 60 |

  The `all` row has maps that are entirely zero after the ReLU, so its pointing rate is not meaningful. Heatmap examples are in `gradcam_randomisation_check.png`; one of three shown cases (fc randomised) still looks similar to the trained map, so this check supports, but does not prove, that the maps depend on the learned weights.

## Measured: pointing game (peak of the heatmap inside any matching box)
Overall: Grad-CAM layer4 38.2% (95% CI 34.0% to 42.7%); layer3 44.0% (95% CI 39.6% to 48.5%); centre prior 17.3% (95% CI 14.1% to 21.0%); permuted heatmaps 15.6% (range of 5 draws 14.5% to 16.7%); random point 2.5% expected.

| Class | n | layer4 | layer3 | centre prior | permuted | random point |
|---|---|---|---|---|---|---|
| Glioma | 169 | 34.3% | 24.3% | 18.3% | 16.7% | 2.6% |
| Meningioma | 153 | 55.6% | 71.9% | 8.5% | 10.2% | 3.6% |
| Pituitary | 146 | 24.7% | 37.7% | 25.3% | 20.1% | 1.3% |

By box size (terciles of box area):

| Box size | n | layer4 | centre prior | random point |
|---|---|---|---|---|
| small boxes (area 0.002 to 0.013 of image) | 156 | 14.1% | 12.8% | 0.8% |
| medium boxes (area 0.013 to 0.027 of image) | 156 | 38.5% | 17.9% | 1.9% |
| large boxes (area 0.027 to 0.144 of image) | 156 | 62.2% | 21.2% | 4.9% |

## Measured: overlap of the hot region (at least 50% of the peak) with the boxes
Mean IoU: layer4 0.120 (95% CI 0.111 to 0.130), layer3 0.141, centre prior 0.082, permuted 0.084. The hot region covers 21.7% of the image on average while the boxes cover 2.5%, which caps the achievable IoU.
Energy enrichment (heat share inside the box divided by the box area share): layer4 median 3.08, centre prior median 2.39, permuted mean 2.49.

## Misclassified test images
8 misclassified tumour images with boxes; peak inside the box in 0 of them (95% CI 0.0% to 32.4%). The sample is too small to compare with correct images.

## Interpretation allowed by these data
- Grad-CAM peaks fall inside the annotated box more often than for the centre prior or for heatmaps borrowed from other images of the same class, so the maps carry image-specific information beyond a central blob. Most peaks (61.8%) are nevertheless outside the box.
- The result differs by class. Meningioma has the highest hit rate of the three classes in both layers. For Pituitary it is not distinguishable from the centre prior (CI overlap), and for Glioma the layer4 map is better than the baselines but the layer3 map is not.
- Hit rates rise with box size, consistent with the coarse 7 x 7 resolution.
- Layer3 and layer4 disagree on which class benefits, so the ranking of the two layers is not stable.

## Not supported
- That the network localises tumours, or bases its decision on the tumour region. The maps are broad, brain-centred regions in most examples, and the boxes are an imperfect reference.
- Any clinical use of the heatmaps.
- A causal reading of heatmaps (no deletion or insertion test was run).

## Limitations and future work
- Single checkpoint, single seed, one test set, one explanation method.
- Boxes of a different class id than the folder (Stage 1) were ignored; two images had none of the matching class.
- Future work: higher-resolution methods (for example Grad-CAM on layer3 combined with layer4, or perturbation-based maps), deletion/insertion tests, and evaluation on an external dataset with tumour masks.
