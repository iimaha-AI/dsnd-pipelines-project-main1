# Customer Review Recommendation Pipeline

A scikit-learn experiment predicting `Recommended IND` from clothing reviews and structured attributes. It combines spaCy preprocessing, TF-IDF, numerical imputation/scaling, categorical encoding, and a random forest.

## Contents

- [starter/starter.ipynb](starter/starter.ipynb): EDA, stratified split, training, GridSearchCV, and evaluation.
- [starter/data/reviews.csv](starter/data/reviews.csv): provided review data.
- `starter/recommendation_model.pkl`: Git LFS model artifact; a pointer is not a downloaded model. Git LFS is required when using the existing artifact.
- [requirements.txt](requirements.txt): original dependency list.

## Setup

```bash
git clone https://github.com/iimaha-AI/dsnd-pipelines-project-main1.git
cd dsnd-pipelines-project-main1
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in PowerShell, then:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd starter
python -m notebook starter.ipynb
```

The dependency list is a starting environment, not a validated lockfile. Known execution limits are listed below.


Run from either the repository root or `starter/`; data and saved-model paths now resolve in both cases. Set `REVIEWS_DATA_PATH` for an external CSV. `OneHotEncoder(sparse_output=False)` requires a supporting scikit-learn version. The grid search has 24 combinations and five folds and may require substantial CPU time/memory.

## Model design and results

Numerical features are age and positive-feedback count. Categorical inputs include clothing ID, division, department, and class. Titles and review bodies use separate TF-IDF transforms. Splitting is stratified with `random_state=42`.

The previous README reported accuracy around 0.86, positive-class F1 around 0.92, and negative-class recall around 0.34. These are historical reported values, not revalidated metrics. Class imbalance makes accuracy and positive-class F1 insufficient descriptions of negative-review detection.

## Next steps

- Lock an environment and data hash after a clean rerun.
- Report majority baseline, macro-F1, per-class recall, and PR-AUC.
- Evaluate whether product-group splitting is necessary.
- `TextProcessor` now lives in `starter/text_processing.py`; keep that module available when loading newly trained artifacts.
- Run the loading/inference smoke check below. See [review notes](docs/REVIEW.md) for earlier findings.

## Attribution

Udacity Data Scientist Nanodegree project. Preserve [LICENSE.txt](LICENSE.txt) and verify the review dataset's source and distribution terms.

## Runtime repair — 2026-09-14

From the repository root, run `python smoke_runtime.py` after installing dependencies and `en_core_web_sm`. The check fits the existing pipeline on 512 stratified records from the supplied CSV with one CPU worker, then compares predictions before and after joblib loading in a fresh Python process. It passed under Python 3.12, pandas 2.3.3, scikit-learn 1.9.1, spaCy 3.8.7 and `en_core_web_sm` 3.8.0.

The transformer’s text-processing logic is unchanged. Old pickles referring to `__main__.TextProcessor` should be regenerated from the notebook; this repair does not rewrite the existing LFS artifact. The full 24-combination, five-fold search was not rerun, so no new benchmark is claimed. `scikit-learn>=1.2,<2` records the required `sparse_output` API. Keep `starter/` on Python’s import path when loading the model outside the notebook.

The existing 95.7 MB LFS artifact was retrieved and its SHA-256 matched the Git LFS pointer. Loading it in the test environment failed inside the serialized spaCy/preshed state with a MemoryError. It was not modified or replaced. Use a newly trained artifact from the corrected notebook, or recover the original training environment before using that old pickle. The successful serialization smoke test concerns a newly fitted 512-row model only.
