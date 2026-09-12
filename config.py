from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
PACS_PATH = PROJECT_ROOT / "data" / "pacs_hf"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SOURCE_MODEL_DIR = OUTPUT_DIR / "source_models"
SOURCE_HISTORY_DIR = OUTPUT_DIR / "source_histories"
SOURCE_PLOT_DIR = OUTPUT_DIR / "source_plots"

ADAPTATION_MODEL_DIR = OUTPUT_DIR / "adapted_models"
ADAPTATION_HISTORY_DIR = OUTPUT_DIR / "adaptation_histories"
ADAPTATION_PLOT_DIR = OUTPUT_DIR / "adaptation_plots"

RESULT_PLOT_DIR = OUTPUT_DIR / "result_plots"
INTERPRETABILITY_DIR = OUTPUT_DIR / "interpretability"

SOURCE_CSV = OUTPUT_DIR / "source_models.csv"
ADAPTATION_CSV = OUTPUT_DIR / "adaptation_results.csv"
CURVE_TARGETS_CSV = OUTPUT_DIR / "curve_targets.csv"
CASE_FEATURES_CSV = OUTPUT_DIR / "case_features.csv"
COLD_START_DATASET_CSV = OUTPUT_DIR / "cold_start_prediction_dataset.csv"
COLD_START_PREDICTIONS_CSV = OUTPUT_DIR / "cold_start_predictions_loto.csv"
COLD_START_FOLD_SUMMARY_CSV = OUTPUT_DIR / "cold_start_fold_summary_loto.csv"
COLD_START_SUMMARY_CSV = OUTPUT_DIR / "cold_start_summary_loto.csv"
WARM_START_DATASET_CSV = OUTPUT_DIR / "warm_start_prediction_dataset.csv"
WARM_START_PREDICTIONS_CSV = OUTPUT_DIR / "warm_start_predictions_loto.csv"
WARM_START_FOLD_SUMMARY_CSV = OUTPUT_DIR / "warm_start_fold_summary_loto.csv"
WARM_START_SUMMARY_CSV = OUTPUT_DIR / "warm_start_summary_loto.csv"

COLD_RIDGE_COEFFICIENTS_CSV = INTERPRETABILITY_DIR / "cold_ridge_coefficients.csv"
COLD_RIDGE_COEFFICIENT_SUMMARY_CSV = INTERPRETABILITY_DIR / "cold_ridge_coefficient_summary.csv"
COLD_RIDGE_PARAMETER_COEFFICIENTS_CSV = INTERPRETABILITY_DIR / "cold_ridge_parameter_coefficients.csv"
COLD_LOFO_ABLATION_CSV = INTERPRETABILITY_DIR / "cold_lofo_ablation.csv"
WARM_RIDGE_COEFFICIENTS_CSV = INTERPRETABILITY_DIR / "warm_ridge_coefficients.csv"
WARM_RIDGE_COEFFICIENT_SUMMARY_CSV = INTERPRETABILITY_DIR / "warm_ridge_coefficient_summary.csv"
WARM_RIDGE_PARAMETER_COEFFICIENTS_CSV = INTERPRETABILITY_DIR / "warm_ridge_parameter_coefficients.csv"
WARM_FEATURE_CORRELATION_MATRIX_CSV = INTERPRETABILITY_DIR / "warm_feature_correlation_matrix.csv"
WARM_FEATURE_CORRELATION_PAIRS_CSV = INTERPRETABILITY_DIR / "warm_feature_correlation_pairs.csv"
WARM_GROUPED_ABLATION_CSV = INTERPRETABILITY_DIR / "warm_grouped_ablation.csv"

SEEDS = [42, 387]
FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
MODEL_NAMES = ["cnn3", "resnet18"]

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 7

MAX_SOURCE_EPOCHS = 100
SOURCE_LR = 1e-3
ADAPTATION_EPOCHS = 100
ADAPTATION_LR = 1e-4

DISTANCE_SAMPLE_LIMIT = 500
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

CURVE_PARAMETER_NAMES = ["curve_a0", "curve_a", "curve_c",]

POWER_LAW_C_MIN = 1e-3
POWER_LAW_C_MAX = 5.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

for directory in [
    OUTPUT_DIR,
    SOURCE_MODEL_DIR,
    SOURCE_HISTORY_DIR,
    SOURCE_PLOT_DIR,
    ADAPTATION_MODEL_DIR,
    ADAPTATION_HISTORY_DIR,
    ADAPTATION_PLOT_DIR,
    RESULT_PLOT_DIR,
    INTERPRETABILITY_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)