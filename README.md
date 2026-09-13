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


Run from `starter/` so `data/reviews.csv` resolves. `OneHotEncoder(sparse_output=False)` requires a supporting scikit-learn version. The grid search has 48 combinations and five folds and may require substantial CPU time/memory.

## Model design and results

Numerical features are age and positive-feedback count. Categorical inputs include clothing ID, division, department, and class. Titles and review bodies use separate TF-IDF transforms. Splitting is stratified with `random_state=42`.

The previous README reported accuracy around 0.86, positive-class F1 around 0.92, and negative-class recall around 0.34. These are historical reported values, not revalidated metrics. Class imbalance makes accuracy and positive-class F1 insufficient descriptions of negative-review detection.

## Next steps

- Lock an environment and data hash after a clean rerun.
- Report majority baseline, macro-F1, per-class recall, and PR-AUC.
- Evaluate whether product-group splitting is necessary.
- Extract `TextProcessor` into an importable module before relying on serialized model portability.
- Add loading/inference tests. See [review notes](docs/REVIEW.md).

## Attribution

Udacity Data Scientist Nanodegree project. Preserve [LICENSE.txt](LICENSE.txt) and verify the review dataset's source and distribution terms.
