# XGBoostUpdated

A from-scratch exploration of XGBoost concepts on the Moon dataset, including a **basic implementation** and an **upgraded v2.0 implementation** with advanced training options.

## Project Contents

- `Upgraded_Versions_Of_Xgboostv.2.ipynb`
  - **Cell 1:** `MyXGBoostClassifier` (v2.0)
    - Histogram binning
    - Multiple losses (`logloss`, `focal`, `quantile`)
    - Leaf-wise (best-first / `lossguide`) and depth-wise growth policies
    - Visualization of decision boundaries
  - **Cell 2:** `BasicXGBoostClassifier`
    - Simpler depth-wise gradient boosted tree baseline
    - Classification on Moon data + decision boundary plot
- `BasicVersion.png` — output/visual comparison image for basic model
- `UpgradedVersion.png` — output/visual comparison image for upgraded model
- `xgboost_deep_dive.docx` — additional notes/documentation

## Key Upgrades in v2.0 vs Basic Version

### New Features Added:
1. **Quantile Regression Loss** ✨
   - Pinball/pseudo-Huber gradient formulation
   - Support for different quantile levels (default 0.5)
   - Useful for regression and robust prediction intervals
   
2. **Leaf-Wise (Best-First) Tree Growth** ✨
   - Controlled by `growth_policy='lossguide'` parameter
   - Mirrors LightGBM's strategy for more efficient tree structure
   - Expands highest-gain leaf first instead of level-wise
   - Can achieve lower loss with fewer leaves on suitable datasets
   
3. **Multiple Loss Functions**
   - `logloss` (traditional binary cross-entropy)
   - `focal` (improved class imbalance handling)
   - `quantile` (NEW - for robustness and prediction intervals)
   
4. **Histogram Binning**
   - Efficient split search with configurable `max_bins` (default 256)
   - Reduces memory usage and speeds up training on large datasets
   - Smooth approximation of exact threshold search
   
5. **Enhanced Hyperparameter Control**
   - `growth_policy`: choose between `'depthwise'` (traditional) or `'lossguide'` (leaf-wise)
   - `num_leaves`: control max leaves per tree in leaf-wise mode
   - `focal_gamma` & `focal_alpha`: focal loss weighting parameters
   - `quantile`: specify target quantile for quantile regression
   - `min_child_weight`: regularization for minimum child node samples

## Requirements

Python 3.9+ recommended.

Install dependencies:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `jupyter`

## Quick Start

1. Open `Upgraded_Versions_Of_Xgboostv.2.ipynb`.
2. Run the first code cell for the upgraded implementation.
3. Run the second code cell for the baseline implementation.
4. Compare metrics/decision boundaries with the included PNG images.

## Example Configuration (Upgraded)

Main hyperparameters used in notebook examples:

- `n_estimators=60`
- `learning_rate=0.15`
- `max_depth=6`
- `num_leaves=31`
- `loss='focal'`
- `growth_policy='lossguide'`
- `max_bins=64`

## Notes

- This repository is educational and implementation-focused (not a drop-in replacement for production `xgboost`).
- For production use, consider the official optimized libraries and GPU-enabled training stacks.
