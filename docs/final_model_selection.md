# Stage 9: Final model selection record

## Decision
**Final model: ResNet18, ImageNet-pretrained, full fine-tuning, run R18-C (seed 42, Stage 6 execution), best-validation checkpoint.**
This is a research model for a student project. It is not a medical device and was not tested on any data outside this dataset.

## Candidates and evidence
| Configuration | runs | trainable params (M) | val macro-F1 | test accuracy | test macro-F1 | test ROC-AUC |
|---|---|---|---|---|---|---|
| CNN from scratch | 1 | 0.98 | 0.9358 | 0.9283 | 0.9238 | 0.9902 |
| CNN, weighted loss | 1 | 0.98 | 0.9547 | 0.9301 | 0.9264 | 0.9909 |
| ResNet18, frozen backbone | 1 | 0.0 | 0.8826 | 0.8768 | 0.8846 | 0.9779 |
| ResNet18, layer4 + fc | 1 | 8.4 | 0.9726 | 0.9559 | 0.9569 | 0.9961 |
| ResNet18, full fine-tuning | 3 | 11.18 | 0.9924 ± 0.0007 | 0.9767 ± 0.0028 | 0.9773 ± 0.0022 | 0.9988 ± 0.0006 |
| ResNet50, frozen backbone | 1 | 0.01 | 0.9127 | 0.8879 | 0.8922 | 0.9816 |
| ResNet50, layer4 + fc | 1 | 14.97 | 0.9830 | 0.9761 | 0.9770 | 0.9988 |
| ResNet50, full fine-tuning | 3 | 23.52 | 0.9906 ± 0.0008 | 0.9792 ± 0.0011 | 0.9794 ± 0.0013 | 0.9995 ± 0.0001 |

Values are mean ± standard deviation over runs when there are three. Test numbers are reported for transparency; they were not used for the choice.

## Rule and reasoning
1. Fine-tuning strategy: full fine-tuning gave the highest validation macro-F1 for both ResNets and was chosen over frozen and layer4-only training.
2. Hyperparameters: the rule fixed before Stage 6 was to adopt a change only if its validation macro-F1 exceeded the reference mean (0.9924) by more than 0.0050 (the larger of the seed range and 0.005). No single-change run met it:

| Run | val macro-F1 | difference from reference mean |
|---|---|---|
| T1-lr1e-5 | 0.9771 | -0.0152 |
| T2-lr1e-3 | 0.9827 | -0.0097 |
| T3-bs16 | 0.9750 | -0.0174 |
| T4-adamw | 0.9819 | -0.0105 |
| T5-dropout | 0.9862 | -0.0062 |
| T6-weighted | 0.9853 | -0.0071 |
| T7-hflip | 0.9947 | +0.0023 |

   Six of the seven variants were lower than the reference. The horizontal-flip variant (T7) was higher by less than the threshold, so it was not adopted. The Stage 4 settings were kept.
3. Architecture: validation macro-F1 means are 0.9924 (ResNet18) and 0.9906 (ResNet50); test macro-F1 means are 0.9773 and 0.9794. The differences are of the same size as the seed-to-seed spread, so the data do not separate the two. ResNet18 was chosen because it has about 48% of the parameters at equivalent measured performance. This is a parsimony choice, not a finding that ResNet18 is better.
4. Seed: among the three ResNet18 seeds, R18-C has the highest validation macro-F1 (0.9932), which is the criterion used.

## What to report as expected performance
The single checkpoint R18-C has test accuracy 0.9798 and macro-F1 0.9797. It is also the best of its three seeds on test, so it is likely at the upper end of what this recipe gives. The mean over three seeds (0.9767 accuracy, 0.9773 macro-F1, standard deviation 0.0028 and 0.0022) is the fairer expectation for a newly trained model. Same-seed reruns of this configuration also differed (R18-C-orig: accuracy 0.9706), so differences of about one percentage point should not be read as real.

## Per-class test results of the final checkpoint
| Class | n | precision | sensitivity (recall) | specificity | F1 |
|---|---|---|---|---|---|
| Glioma | 169 | 0.977 | 0.988 (CI 0.958 to 0.997) | 0.989 | 0.982 |
| Meningioma | 155 | 0.987 | 0.955 (CI 0.910 to 0.978) | 0.995 | 0.971 |
| No Tumor | 74 | 0.986 | 0.973 (CI 0.907 to 0.993) | 0.998 | 0.980 |
| Pituitary | 146 | 0.973 | 1.000 (CI 0.974 to 1.000) | 0.990 | 0.987 |

## Known limitations to state with the model
- No patient identifiers: possible same-patient images across splits and source heterogeneity could not be separated (Stage 7). The test accuracy may be optimistic; an external test set is needed.
- Errors are concentrated in Meningioma and in images from a different filename family (a proxy for source), and errors tend to occur on images less similar to any training image.
- High-confidence errors occur, so the predicted probability is not a reliable warning.
- Grad-CAM agreement with the boxes is weak (Stage 8), so the heatmaps must not be presented as tumour localisation.
- Four classes only; other tumour types, other sequences, other scanners and post-operative images were not covered.
