# IGMT

Code framework for **Inconsistency-Guided Model Transfer for State-of-Health Estimation of Lithium-Ion Battery Packs**.

[中文说明](README_zh.md) · [Method-to-code map](docs/method_map.md) · [Input format](docs/data_format.md)

The code covers all three components of Section II. It contains the method itself, without comparison methods, ablation experiments or figure-generation scripts.

1. **Quantification of Cell Inconsistency in Battery Packs.** Median-centered voltage/temperature deviations, scaled MAD, top-deviation aggregation, dynamic divergence, source-domain normalization and CRITIC fusion.
2. **Measurement of Discrepancies in Inconsistency.** Operating-condition partitions and overlap, quantile Wasserstein-1 distance, trailing dynamic windows, standardization/PCA/RFF, debiased MMD², and source-calibrated discrepancy fusion.
3. **Transfer Learning Guided by Inconsistency.** Shared LSTM encoding, source pretraining, local prediction-head adaptation with L2-SP, regression-forest suitability scoring, and selection between a local model and the pretrained model.

## Structure

```text
igmt/
  framework/
    types.py              Event and inconsistency-state interfaces
    representation.py     Section II-A, Eqs. (1)–(5)
    discrepancy.py        Section II-B, Eqs. (6)–(9)
    model.py              Shared encoder and prediction head, Eqs. (10)–(11)
    transfer.py           Pretraining, local adaptation and selection, Eqs. (11)–(13)
    data.py               Source-fitted preprocessing and event input
    pipeline.py           Unified IGMTFramework.fit / predict / save / load
    __main__.py           Command-line entry
  metrics.py              SOH evaluation utilities
configs/                  Representation, discrepancy and transfer parameters
docs/                     Formula mapping and input instructions
```

## Workflow

```text
Cell voltage and temperature
    -> G/W/D indicators -> common-scale representation u(t), index I(t)
    -> condition overlap -> Wasserstein-1 + PCA/RFF/MMD² -> fused discrepancy

Labeled source training events -> shared LSTM + baseline-correction head
Labeled local source events   -> local prediction heads with L2-SP
Independent source holdout    -> forest learns expected error reduction
Unlabeled target events      -> discrepancy-guided expert selection -> SOH
```

The default encoder concatenates the final LSTM state, masked mean state, input mean/SD and auxiliary information. Its representation has `64 + 64 + 30 + 16 = 174` dimensions. Local adaptation updates the projection and output layers while retaining the shared encoder. The selection target is `|base prediction − reference| − |local prediction − reference|`, evaluated only on held-out source packs. A local model is selected only if its predicted improvement is positive.

## Use

Requires Python 3.11+, NumPy, PyTorch, scikit-learn and joblib.

```bash
python -m pip install -e .
python -m igmt.framework fit --pretrain data/source_train --local-source data/source_local --selector-source data/source_holdout --config configs --output outputs/igmt.joblib
python -m igmt.framework predict --model outputs/igmt.joblib --target data/target --output outputs/predictions.csv
```

Each input directory contains event NPZ files following [the data contract](docs/data_format.md). The four pack roles are separate. Target SOH labels are not read during prediction. The source-trained preprocessing and local models are reused for new target packs.

The Python entry point is `igmt.framework.IGMTFramework`. Its `fit` method constructs the offline model library and selector; `predict` performs target estimation without retraining a complete model.

## Revision status

This revision completes the implementation according to the manuscript's method architecture. **At the author's request, this revision was not run, trained or tested, and no experimental results were generated.** Parameters not fixed in the manuscript are explicit in `configs/`; they are implementation defaults, not reported experimental settings.

Raw datasets and trained weights are not included. The folder can be uploaded as the repository root. See [upload instructions](docs/github_upload.md).
