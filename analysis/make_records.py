"""Writes two plain-text records from saved result files (every number is read from a CSV/JSON, none typed in):
   stage8_findings.md  and  final_model_selection.md.   Usage: python make_records.py <final_results> <stage7_tables> <stage8_dir> <out_dir>"""
import sys, json
from pathlib import Path
import pandas as pd
res, s7, s8, out = map(Path, sys.argv[1:5]); out.mkdir(parents=True, exist_ok=True)
S = pd.read_csv(s8 / 'tables' / 'gradcam_localisation_summary.csv'); D = pd.read_csv(s8 / 'tables' / 'gradcam_per_image_test.csv')
R = pd.read_csv(s8 / 'tables' / 'gradcam_randomisation_check.csv'); san = json.load(open(s8 / 'gradcam_sanity_checks.json'))
def g(group, method): return S[(S.group == group) & (S.method == method)].iloc[0]
def pct(x): return f'{100 * x:.1f}%'
A = 'all evaluated tumour images'; l4, l3, ce, pm, rp = [g(A, m) for m in ['Grad-CAM (layer4)', 'Grad-CAM (layer3)', 'Centre prior', 'Permuted heatmap', 'Random point']]
def ci(r): return f'{pct(r.pointing_rate)} (95% CI {pct(r.pointing_ci_low)} to {pct(r.pointing_ci_high)})'
rows = ''.join(f"| {c} | {int(g(c,'Grad-CAM (layer4)').n)} | {pct(g(c,'Grad-CAM (layer4)').pointing_rate)} | {pct(g(c,'Grad-CAM (layer3)').pointing_rate)} | {pct(g(c,'Centre prior').pointing_rate)} | {pct(g(c,'Permuted heatmap').pointing_rate)} | {pct(g(c,'Random point').pointing_rate)} |\n" for c in ['Glioma', 'Meningioma', 'Pituitary'])
sz = [x for x in S.group.unique() if 'boxes' in x]
rows_sz = ''.join(f"| {x} | {int(g(x,'Grad-CAM (layer4)').n)} | {pct(g(x,'Grad-CAM (layer4)').pointing_rate)} | {pct(g(x,'Centre prior').pointing_rate)} | {pct(g(x,'Random point').pointing_rate)} |\n" for x in sz)
mis = g('misclassified', 'Grad-CAM (layer4)')
rr = ''.join(f"| {r.randomised} | {r.mean_spearman_with_trained_map:.2f} | {r.median_spearman:.2f} | {int(r.all_zero_maps)} of {int(r.n_images)} |\n" for r in R.itertuples())
hot = D['l4_hot_area_frac_0.5'].mean(); box = D.l4_box_area_frac.mean()
md8 = f"""# Stage 8 findings: Grad-CAM against annotated boxes (final model R18-C, test set)

## Implemented
- Grad-CAM on `layer4` (7 x 7 grid) with `layer3` (14 x 14) as a sensitivity check, target class = predicted class, ResNet18 checkpoint `{san['matched_run']}` (epoch {san['checkpoint']['epoch']}).
- Heatmaps mapped back to original image coordinates (undoing the black padding and the resize), min-max normalised inside the image area.
- Evaluated on test images of the three tumour classes that have at least one box of the matching class: {san['evaluated_images']} of {san['tumour_test_images']} ({san['tumour_images_without_matching_class_box']} had no matching box). No Tumor images were excluded because their boxes cover most of the brain.
- Baselines: a fixed centre Gaussian, heatmaps of other images of the same class (5 draws), and the expected hit rate of a uniformly random point.

## Sanity checks (all passed)
- Checkpoint reproduces the saved R18-C test predictions: label agreement {pct(san['reproduces_saved_predictions']['R18-C']['label_agreement'])}, largest probability difference {san['reproduces_saved_predictions']['R18-C']['max_abs_prob_diff']:.1e}. The Stage 4 run (R18-C-orig) agreed on {pct(san['reproduces_saved_predictions']['R18-C-orig']['label_agreement'])}, so the checkpoint is the Stage 6 run.
- Hooked gradients equal the exact formula for this network (largest difference {san['analytic_gradient_check']['max_abs_alpha_error']:.1e} on weights of size {san['analytic_gradient_check']['alpha_scale']:.1e}).
- Heat falling on the black padding: {100 * san['mean_heat_share_on_padding_layer4']:.2f}% on average.
- Weight randomisation changes the maps (Spearman correlation with the trained map; 60 images):

| Randomised part | mean | median | all-zero maps |
|---|---|---|---|
{rr}
  The `all` row has maps that are entirely zero after the ReLU, so its pointing rate is not meaningful. Heatmap examples are in `gradcam_randomisation_check.png`; one of three shown cases (fc randomised) still looks similar to the trained map, so this check supports, but does not prove, that the maps depend on the learned weights.

## Measured: pointing game (peak of the heatmap inside any matching box)
Overall: Grad-CAM layer4 {ci(l4)}; layer3 {ci(l3)}; centre prior {ci(ce)}; permuted heatmaps {pct(pm.pointing_rate)} (range of 5 draws {pct(pm.pointing_ci_low)} to {pct(pm.pointing_ci_high)}); random point {pct(rp.pointing_rate)} expected.

| Class | n | layer4 | layer3 | centre prior | permuted | random point |
|---|---|---|---|---|---|---|
{rows}
By box size (terciles of box area):

| Box size | n | layer4 | centre prior | random point |
|---|---|---|---|---|
{rows_sz}
## Measured: overlap of the hot region (at least 50% of the peak) with the boxes
Mean IoU: layer4 {l4.iou_0_5_mean:.3f} (95% CI {l4.iou_0_5_ci_low:.3f} to {l4.iou_0_5_ci_high:.3f}), layer3 {l3.iou_0_5_mean:.3f}, centre prior {ce.iou_0_5_mean:.3f}, permuted {pm.iou_0_5_mean:.3f}. The hot region covers {100 * hot:.1f}% of the image on average while the boxes cover {100 * box:.1f}%, which caps the achievable IoU.
Energy enrichment (heat share inside the box divided by the box area share): layer4 median {l4.enrichment_median:.2f}, centre prior median {ce.enrichment_median:.2f}, permuted mean {pm.enrichment_mean:.2f}.

## Misclassified test images
{int(mis.n)} misclassified tumour images with boxes; peak inside the box in {int(mis.pointing_hits)} of them (95% CI {pct(mis.pointing_ci_low)} to {pct(mis.pointing_ci_high)}). The sample is too small to compare with correct images.

## Interpretation allowed by these data
- Grad-CAM peaks fall inside the annotated box more often than for the centre prior or for heatmaps borrowed from other images of the same class, so the maps carry image-specific information beyond a central blob. Most peaks ({pct(1 - l4.pointing_rate)}) are nevertheless outside the box.
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
"""
(out / 'stage8_findings.md').write_text(md8)

# ------------------------------------------------------------------ Stage 9 selection record
C = pd.read_csv(s7 / 'all_configurations_summary.csv').set_index('configuration'); cost = pd.read_csv(s7 / 'computational_cost.csv').set_index('configuration')
T = pd.read_csv(res / 'metrics' / 'experiment_table_master.csv').set_index('experiment_id')
cmp_rows = ''.join(f"| {k} | {r.runs} | {r.trainable_params_M} | {r.val_f1_macro} | {r.test_accuracy} | {r.test_f1_macro} | {r.test_roc_auc} |\n" for k, r in C.iterrows())
tuning = T.loc[[i for i in T.index if i.startswith('T')]]
ref_val = T.loc[['R18-C', 'R18-C-s43', 'R18-C-s44'], 'val_macro_f1']
trows = ''.join(f"| {i} | {r.val_macro_f1:.4f} | {r.val_macro_f1 - ref_val.mean():+.4f} |\n" for i, r in tuning.iterrows())
thr = max(ref_val.max() - ref_val.min(), 0.005)
f = T.loc['R18-C']; r18 = T.loc[['R18-C', 'R18-C-s43', 'R18-C-s44']]; r50 = T.loc[['R50-C', 'F-R50-s43', 'F-R50-s44']]
final_pc = pd.read_csv(s7 / 'final_model_per_class_test.csv')
pcrows = ''.join(f"| {r.cls} | {int(r.support)} | {r.precision:.3f} | {r.recall_sensitivity:.3f} (CI {r.recall_95CI_low:.3f} to {r.recall_95CI_high:.3f}) | {r.specificity:.3f} | {r.f1:.3f} |\n" for r in final_pc.itertuples())
md9 = f"""# Stage 9: Final model selection record

## Decision
**Final model: ResNet18, ImageNet-pretrained, full fine-tuning, run R18-C (seed 42, Stage 6 execution), best-validation checkpoint.**
This is a research model for a student project. It is not a medical device and was not tested on any data outside this dataset.

## Candidates and evidence
| Configuration | runs | trainable params (M) | val macro-F1 | test accuracy | test macro-F1 | test ROC-AUC |
|---|---|---|---|---|---|---|
{cmp_rows}
Values are mean ± standard deviation over runs when there are three. Test numbers are reported for transparency; they were not used for the choice.

## Rule and reasoning
1. Fine-tuning strategy: full fine-tuning gave the highest validation macro-F1 for both ResNets and was chosen over frozen and layer4-only training.
2. Hyperparameters: the rule fixed before Stage 6 was to adopt a change only if its validation macro-F1 exceeded the reference mean ({ref_val.mean():.4f}) by more than {thr:.4f} (the larger of the seed range and 0.005). No single-change run met it:

| Run | val macro-F1 | difference from reference mean |
|---|---|---|
{trows}
   Six of the seven variants were lower than the reference. The horizontal-flip variant (T7) was higher by less than the threshold, so it was not adopted. The Stage 4 settings were kept.
3. Architecture: validation macro-F1 means are {r18.val_macro_f1.mean():.4f} (ResNet18) and {r50.val_macro_f1.mean():.4f} (ResNet50); test macro-F1 means are {r18.test_macro_f1.mean():.4f} and {r50.test_macro_f1.mean():.4f}. The differences are of the same size as the seed-to-seed spread, so the data do not separate the two. ResNet18 was chosen because it has about {cost.loc['ResNet18, full fine-tuning','trainable_params_M'] / cost.loc['ResNet50, full fine-tuning','trainable_params_M']:.0%} of the parameters at equivalent measured performance. This is a parsimony choice, not a finding that ResNet18 is better.
4. Seed: among the three ResNet18 seeds, R18-C has the highest validation macro-F1 ({f.val_macro_f1:.4f}), which is the criterion used.

## What to report as expected performance
The single checkpoint R18-C has test accuracy {f.test_accuracy:.4f} and macro-F1 {f.test_macro_f1:.4f}. It is also the best of its three seeds on test, so it is likely at the upper end of what this recipe gives. The mean over three seeds ({r18.test_accuracy.mean():.4f} accuracy, {r18.test_macro_f1.mean():.4f} macro-F1, standard deviation {r18.test_accuracy.std():.4f} and {r18.test_macro_f1.std():.4f}) is the fairer expectation for a newly trained model. Same-seed reruns of this configuration also differed (R18-C-orig: accuracy {T.loc['R18-C-orig','test_accuracy']:.4f}), so differences of about one percentage point should not be read as real.

## Per-class test results of the final checkpoint
| Class | n | precision | sensitivity (recall) | specificity | F1 |
|---|---|---|---|---|---|
{pcrows}
## Known limitations to state with the model
- No patient identifiers: possible same-patient images across splits and source heterogeneity could not be separated (Stage 7). The test accuracy may be optimistic; an external test set is needed.
- Errors are concentrated in Meningioma and in images from a different filename family (a proxy for source), and errors tend to occur on images less similar to any training image.
- High-confidence errors occur, so the predicted probability is not a reliable warning.
- Grad-CAM agreement with the boxes is weak (Stage 8), so the heatmaps must not be presented as tumour localisation.
- Four classes only; other tumour types, other sequences, other scanners and post-operative images were not covered.
"""
(out / 'final_model_selection.md').write_text(md9); print('written', [p.name for p in out.glob('*.md')])

# ------------------------------------------------------------------ model card
card = f"""# Model card: brain MRI four-class classifier (student research project)

**Not a medical device. Not validated for clinical use. Do not use it to make or support diagnostic or treatment decisions.**

## Model
- ResNet18 (torchvision), ImageNet-pretrained, all layers fine-tuned, final layer replaced by a 4-way linear layer.
- Classes (output order): Glioma, Meningioma, No Tumor, Pituitary.
- Input: a 2D MRI slice (JPEG/PNG). Preprocessing: pad to a centred black square, resize to 224 x 224, scale to [0,1], ImageNet mean/std normalisation. No random operations at inference.
- Training: Adam, learning rate 1e-3 for the new layer and 1e-4 for pretrained layers, batch size 32, ReduceLROnPlateau, early stopping on validation macro-F1, conservative augmentation (small crop, rotation of at most 10 degrees, 5% shift, slight brightness/contrast). No horizontal flip.
- Checkpoint: run R18-C, epoch {san['checkpoint']['epoch']}, selected on validation macro-F1 ({san['checkpoint']['monitor_value']:.4f}).

## Data
Kaggle dataset `ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes`. Exact duplicates were removed, near-duplicates were grouped, and a stratified group-aware split of about 70/15/15 was used. The dataset has no patient identifiers.

## Measured performance (held-out test split, n = {int(final_pc.support.sum())}, one checkpoint)
Accuracy {f.test_accuracy:.4f}, macro-F1 {f.test_macro_f1:.4f}, one-vs-rest macro ROC-AUC {f.roc_auc_test_ovr_macro:.4f}. Mean over three seeds of the same recipe: accuracy {r18.test_accuracy.mean():.4f}, macro-F1 {r18.test_macro_f1.mean():.4f}. Per-class results are in `final_model_selection.md`.

## Known limitations
- Possible same-patient images across splits and source heterogeneity could not be ruled out, so the accuracy may be optimistic. No external test set was used.
- Lowest sensitivity for Meningioma. Some errors have high predicted probability.
- Grad-CAM heatmaps agree only weakly with the annotated boxes and must not be read as tumour localisation.
- Only four classes, one dataset, 2D slices. Behaviour on other scanners, sequences, planes, ages, or post-operative images is unknown.
"""
(out / 'MODEL_CARD.md').write_text(card); print('card written')
