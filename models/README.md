# Model checkpoint

The trained checkpoint `R18-C_best.pt` (about 45 MB) is not committed by default (`*.pt` is in `.gitignore`).
Place it here to run `analysis/predict.py`. Its SHA-256 checksum is written to `final_model_metadata.json` by `notebooks/09_final_model.ipynb`.
To share it, attach it to a GitHub Release or remove `*.pt` from `.gitignore` (the file is below GitHub's 100 MB limit).
