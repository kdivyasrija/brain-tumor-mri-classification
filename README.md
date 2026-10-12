# Brain tumour MRI classification with transfer learning: an evaluation-focused student project

Four-class classification of brain MRI slices (Glioma, Meningioma, Pituitary tumour, No Tumor) with a small CNN and ImageNet-pretrained ResNet18 and ResNet50, together with a data audit, a leakage-checked split, repeated runs, error analysis and a Grad-CAM comparison with annotated bounding boxes.

> **Research and teaching use only.** This is a student project. The model is not a medical device, has not been validated on any data outside the dataset below, and must not be used to diagnose or to support treatment decisions.

## What was found (all numbers are computed from the files in `results/`)

| Model | Runs | Test accuracy | Test macro-F1 |
|---|---|---|---|
| CNN from scratch (B1) | 1 | 0.9283 | 0.9238 |
| ResNet18, frozen backbone | 1 | 0.8768 | 0.8846 |
| ResNet18, full fine-tuning | 3 seeds | 0.9767 (sd 0.0028) | 0.9773 (sd 0.0022) |
| ResNet50, full fine-tuning | 3 seeds | 0.9792 (sd 0.0011) | 0.9794 (sd 0.0013) |
| **Final model: ResNet18 full fine-tuning, run R18-C** | 1 checkpoint | 0.9798 | 0.9797 |

* Full fine-tuning beat a frozen backbone and the scratch CNN. ResNet18 and ResNet50 are not separable within seed variation, so ResNet18 was chosen for its smaller size.
* None of seven one-factor hyperparameter changes passed a decision rule fixed in advance.
* Two executions of the same configuration and seed gave different results (`R18-C-orig` and `R18-C`), so differences of about one percentage point between single runs are not interpretable.
* The supplied `Val` folder of the dataset leaks into `Train` (218 of 512 images are byte-identical to training images), and 48.8 percent of all images are in exact-duplicate groups. A cleaned, group-aware split was built instead.
* Errors concentrate on Meningioma, on certain filename families (a proxy for image source) and on images that are less similar to any training image. The dataset has no patient identifiers, so same-patient leakage and source differences cannot be separated. **The test accuracy may be optimistic for new patients and scanners.**
* Grad-CAM peaks fall inside the annotated tumour box for 38.2% of tumour test images (centre-prior baseline 17.3%). The heatmaps agree with the boxes only weakly and are **not** evidence that the network localises tumours.

Full discussion: see the report (`report/`) and `docs/`. Repository: https://github.com/kdivyasrija/brain-tumor-mri-classification

## Repository layout

```
notebooks/   Colab notebooks 01 to 09, one per project stage (run in order)
src/         model, data, training, evaluation, plotting and Grad-CAM code
analysis/    scripts used by the Stage 7 to 9 notebooks (merge results, final evaluation, Grad-CAM analysis, inference script)
tests/       self-tests of the coordinate mapping and localisation metrics (no PyTorch needed)
results/     experiment table, per-run metrics and predictions, Stage 1 audit, Stage 7 tables and figures, Stage 8 tables and figures
docs/        selection record, Stage 8 findings, model card, reference verification list
models/      place the trained checkpoint here (see models/README.md)
report/      the project report (Word file)
```

## Stages and notebooks

| Notebook | Stage |
|---|---|
| `01_dataset_analysis.ipynb` | Dataset inspection and audit (duplicates, labels, boxes, shortcuts) |
| `02_preprocessing.ipynb` | Cleaning, near-duplicate grouping, stratified group-aware split, loaders |
| `03_baseline_cnn.ipynb` | Baseline CNN (B1, B2) |
| `04_resnet18.ipynb`, `05_resnet50.ipynb` | Transfer-learning strategies A, B, C |
| `06a_hyperparameter_tuning.ipynb`, `06b_seed_repeats.ipynb` | One-factor experiments and seed repeats |
| `07_evaluation.ipynb` | Final evaluation and error analysis (reads saved results, no GPU) |
| `08_gradcam.ipynb` | Grad-CAM and bounding-box comparison (GPU) |
| `09_final_model.ipynb` | Final model package and check of the inference script |

## Reproducing the work

1. **Data.** Download the Kaggle dataset `ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes` yourself (the images are not part of this repository; check the dataset page for its licence). In Colab, store your own Kaggle credentials as Colab secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`) or upload the dataset zip to Google Drive. **Never commit credentials.**
2. **Notebooks.** Open the notebooks in Google Colab with a GPU runtime and run them in order. Each notebook writes its own copy of the code in `src/`. Notebooks 04 to 06b back up to Google Drive and can resume after a disconnect.
3. **Results.** `results/` contains the outputs of the runs reported in the paper. `results/experiment_table_master.csv` lists all 20 runs. Run `R18-C-orig` is the first execution of the ResNet18 full fine-tuning setting; `R18-C` is the later execution used as the final model.
4. **Tests.** `python tests/test_localisation.py` (needs only NumPy, SciPy and Pillow, with `src/localisation.py` on the path).
5. **Inference.** `python analysis/predict.py --ckpt models/R18-C_best.pt --images scan.jpg`. This is a research prototype; see `docs/MODEL_CARD.md`.

Environment of the reported runs: Google Colab, Tesla T4, PyTorch 2.11.0 (CUDA 13.0 build), Python 3.13. See `requirements.txt` for the packages.

## Notes on reproducibility

* Seeds 42, 43 and 44 were used. cuDNN deterministic mode was requested, yet repeated runs of the same configuration still differed.
* `src/` is the final version used in Stage 6. Earlier notebooks embed their own versions of these files, which they write at run time.
* The model checkpoint is not stored in this repository by default. Its SHA-256 checksum is recorded in `final_model_metadata.json` created by notebook 09.

## Limitations

No patient identifiers; possible same-patient leakage and source effects; one dataset and one test split; few seeds; weak agreement of heatmaps with boxes; two-dimensional slices only; not validated for clinical use.

## Licence

Code: MIT licence (see `LICENSE`). The dataset has its own licence and is not redistributed here.

## Author

[Student name], B.Tech Bioinformatics, [Institute]. Guide: [Guide name].
