# Results Summary

## Dataset

- Raw dataset size: 2017 tracks
- Tracks used for modeling after dropping missing `track_popularity`: 1882
- Hit threshold: top 25% of `track_popularity`
- Numeric target cutoff from saved run: `62.0`

## Model Performance

From `results/model_output.txt`:

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Dummy baseline | 0.746 | N/A | N/A | N/A |
| Logistic Regression | 0.851 | 0.733 | 0.656 | 0.692 |
| SVM (RBF) | 0.870 | 0.790 | 0.667 | 0.723 |

## Main Takeaways

- The SVM with an RBF kernel is the strongest model in the current repo.
- `artist_popularity` is the most important predictor by a wide margin.
- Audio features like `valence`, `energy`, `danceability`, and `loudness` still contribute useful signal.
- Recall is good enough to show real learning, but improving it is the clearest next modeling step.

## Figures

- `figures/roc_logreg_vs_svm.png`: ROC comparison between Logistic Regression and SVM
- `figures/perm_importance_svm.png`: Top permutation importance scores for the SVM
- `figures/violin_*.png`: Distribution comparisons for selected audio features by hit label

