# PACS Learning-Curve Prediction

A compact pilot study on forecasting how target-domain performance changes as labeled target data become available.

The project uses the PACS domain-shift benchmark to study two prediction settings:

- **Cold start:** predict the full target learning curve before any target adaptation training.
- **Warm start:** after observing the 5% and 10% adaptation stages, predict performance at the remaining 25%, 50%, and 100% target-data budgets.

The experiment is intended as a proof of concept for a broader research direction on **predicting data sufficiency for model adaptation from model-, data-, and early-training signals**. It is not presented as a benchmark result or as evidence of universal learning-curve behavior.

## Main idea

For each source-to-target case, the measured target-test learning curve is summarized by

$$
A(n)=A_0+(a-A_0)\left[1-(n+1)^{-c}\right],
$$

where:

- $n$ is the exact number of labeled target-training examples,
- $A_0$ is zero-shot target-test accuracy,
- $a$ is a bounded asymptotic parameter,
- $c$ controls the rate of improvement.

The fitted parameters $[A_0,a,c]$ become supervised targets for Ridge predictors. Prediction quality is evaluated after reconstructing the curve in accuracy space at the exact case-specific sample counts.

## Experimental setup

PACS contains four domains: `art_painting`, `cartoon`, `photo`, and `sketch`, with seven classes.

Each domain is split deterministically into:

- 70% training
- 10% validation
- 20% test

Source models are trained from scratch on every one-domain and two-domain source combination using:

- a compact three-layer CNN (`cnn3`)
- a randomly initialized ResNet-18 (`resnet18`)
- two seeds: `42` and `387`

This produces 40 source checkpoints and 96 valid source-to-target cases in which the target domain is absent from the source domains.

Each source checkpoint is independently adapted using nested target-training subsets corresponding nominally to 5%, 10%, 25%, 50%, and 100% of the target training pool. Exact sample counts are retained throughout the analysis.

## Evaluation

The meta-predictors use **leave-one-target-domain-out** outer evaluation. In each outer fold, all cases targeting one PACS domain are held out.

Ridge regularization is selected inside the outer training set with grouped cross-validation, where rows sharing the same source checkpoint (`source_id`) remain in the same inner fold.

This separates two roles:

- the outer split evaluates generalization to an unseen target domain;
- the inner grouped split selects Ridge regularization without splitting closely related cases from the same source checkpoint across inner train and validation sets.

## Predictors

### Cold start

The main metadata Ridge uses:

- model architecture
- source training-set size
- first target-subset size
- source validation accuracy
- zero-shot target validation accuracy
- zero-shot target validation loss

Additional variants use global source-target representation discrepancies measured with the original source model on the 5% target subset:

- RBF-MMD squared
- centroid Euclidean distance
- centroid cosine distance
- normalized covariance Frobenius distance

Zero-shot target **test** accuracy is never used as a Ridge input.

### Warm start

Warm start adds information from the 5% and 10% adaptation runs:

- exact target sample counts
- best epoch
- best training accuracy
- best validation accuracy
- optimization-step count

Representation-discrepancy variants use the 10% target subset and compare discrepancies under the original source checkpoint, the 10%-adapted checkpoint, and their signed change.

A simple three-point logarithmic extrapolation using zero-shot, 5%, and 10% **validation** accuracies is included as a warm-start baseline.

## Results

Across 96 learning-curve cases, the parametric curve family fits the six measured target-test points with mean MAE **0.0287**.

### Cold start

| Predictor | Overall MAE |
| --- | ---: |
| Metadata Ridge | **0.0896** |
| Metadata + discrepancy Ridge | 0.1102 |
| Mean curve | 0.1212 |
| Discrepancy Ridge | 0.1233 |

The tested global discrepancy summaries do not improve the strongest cold-start predictor.

Cold-start leave-one-feature-out analysis identifies zero-shot target validation accuracy as the clearest individual metadata signal: removing it increases MAE from 0.0896 to 0.1266.

### Warm start

| Predictor | Overall MAE |
| --- | ---: |
| Three-point log curve | **0.0521** |
| Metadata + observation Ridge | 0.0600 |
| Metadata + observation + discrepancy change Ridge | 0.0686 |
| Observation Ridge | 0.0711 |
| Mean remaining curve | 0.1283 |

The three-point logarithmic baseline is the strongest warm-start method in this pilot.

The warm feature set contains substantial redundancy. Examples include correlations of approximately:

- `n_5` vs. `n_10`: 0.99999
- `n_5` vs. `optimization_steps_10`: 0.9943
- `best_val_accuracy_5` vs. `best_val_accuracy_10`: 0.9707

Because single-feature ablations can be misleading under this level of collinearity, the warm-start analysis uses grouped feature ablation.

| Removed warm-start information group | MAE | Change vs. full |
| --- | ---: | ---: |
| None | 0.0600 | 0.0000 |
| Early validation performance | 0.0827 | +0.0227 |
| Source / zero-shot context | 0.0686 | +0.0086 |
| Budget / training amount | 0.0562 | -0.0038 |
| Training dynamics | 0.0550 | -0.0050 |

The clearest incremental warm-start information comes from early validation performance. The budget/training-amount and training-dynamics groups appear largely redundant or noisy conditional on the remaining inputs.

## Repository structure

```text
.
├── config.py
├── data.py
├── models.py
├── training.py
├── metrics.py
├── prediction.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── report.pdf
├── stages/
│   ├── 01_train_sources.py
│   ├── 02_run_adaptation.py
│   ├── 03_build_curve_targets.py
│   ├── 04_extract_cold_start_features.py
│   ├── 05_evaluate_cold_start.py
│   ├── 06_build_warm_start_dataset.py
│   ├── 07_evaluate_warm_start.py
│   └── 08_analyze_interpretability.py
├── notebooks/
│   └── results_and_figures.ipynb
└── outputs/
    ├── *.csv
    ├── interpretability/
    │   └── *.csv
    └── result_plots/
        └── *.png
```

## Reproducing the experiment

### 1. Environment

The reported results were produced with **Python 3.12.7**.

Install the pinned dependencies with:

```bash
pip install -r requirements.txt
```

The provided `requirements.txt` records the exact package versions used for the final experiment, including the CUDA 12.1 PyTorch build.

### 2. Prepare PACS

The code expects PACS to be stored in Hugging Face `load_from_disk` format at:

```text
data/pacs_hf
```

The loaded dataset must contain at least the following fields:

- `image`
- `label`
- `domain`

The `domain` field should identify the four PACS domains:

```text
art_painting
cartoon
photo
sketch
```

Once the dataset is available at `data/pacs_hf`, no manual split files are required. The train/validation/test splits and nested target subsets are created deterministically by `data.py`.

### 3. Run the pipeline

Run the stages sequentially from the project root:

```bash
python stages/01_train_sources.py
python stages/02_run_adaptation.py
python stages/03_build_curve_targets.py
python stages/04_extract_cold_start_features.py
python stages/05_evaluate_cold_start.py
python stages/06_build_warm_start_dataset.py
python stages/07_evaluate_warm_start.py
python stages/08_analyze_interpretability.py
```

Stages 1 and 2 save intermediate model checkpoints, histories, and training plots locally under `outputs/`. They also contain resume logic so completed runs are not needlessly repeated when their expected artifacts are still present.

Running the full pipeline regenerates the compact CSV results, final evaluation plots, and interpretability outputs used in the report and notebook.

### 4. Inspect the analysis

After Stages 1–8 have completed, open:

```text
notebooks/results_and_figures.ipynb
```

The notebook is deliberately **analysis-only**: it loads the completed experiment outputs and creates the final tables and figures. It does not train, fine-tune, or refit the prediction models.

## Included outputs

The main compact experiment outputs include:

```text
outputs/curve_targets.csv
outputs/case_features.csv

outputs/cold_start_prediction_dataset.csv
outputs/cold_start_predictions_loto.csv
outputs/cold_start_fold_summary_loto.csv
outputs/cold_start_summary_loto.csv

outputs/warm_start_prediction_dataset.csv
outputs/warm_start_predictions_loto.csv
outputs/warm_start_fold_summary_loto.csv
outputs/warm_start_summary_loto.csv
```

Stage 8 writes the final interpretability outputs to:

```text
outputs/interpretability/
```

including:

- cold Ridge coefficient summaries
- cold Ridge coefficient details
- cold leave-one-feature-out ablation
- warm Ridge coefficient summaries
- warm Ridge coefficient details
- the warm feature-correlation matrix and ranked correlation pairs
- warm grouped ablation results

Final comparison plots are stored in:

```text
outputs/result_plots/
```

## Reproducibility notes

- PACS train/validation/test splits use a fixed random state of 42.
- Target adaptation subsets are deterministic and nested.
- Source-model seeds are 42 and 387.
- Neural-network checkpoints are selected by validation accuracy.
- Ground-truth learning curves are defined from held-out target-test accuracies.
- Ridge alpha is selected separately inside each outer fold using grouped inner cross-validation.
- Final reported prediction errors are computed in accuracy space rather than directly on curve parameters.
- Exact package versions used for the reported results are recorded in `requirements.txt`.

## Limitations

This is intentionally a small pilot. It uses four PACS domains, two model architectures, two seeds, one supervised adaptation procedure, a simple three-parameter learning-curve family, and linear Ridge predictors.

The fitted asymptotic parameter reaches its upper bound in 95 of 96 cases, so the parametric curves should be interpreted mainly over the measured adaptation range rather than as reliable long-range asymptotic estimates.

The main purpose of the experiment is to demonstrate a reproducible framework for asking whether future adaptation performance—and eventually data sufficiency—can be forecast from signals available before or shortly after adaptation begins.

## Report

A compact PDF report is included as `report.pdf`. It describes the motivation, methodology, evaluation protocol, results, interpretability analysis, and limitations in more detail.

## Author

**Emre Çağan Kanlı**
