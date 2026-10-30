from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
TRAIN_FILE = RAW_DIR / "KDDTrain+.txt"
TEST_FILE = RAW_DIR / "KDDTest+.txt"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"
PRED_DIR = RESULTS_DIR / "predictions"
TABLE_DIR = RESULTS_DIR / "tables"
FIG_DIR = RESULTS_DIR / "figures"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.joblib"

SPLIT_SEED = 42
VAL_FRACTION = 0.15
SEEDS = [0, 1, 2, 3, 4]

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
PATIENCE = 3

HIDDEN = 64
TIME_STEPS = 25
BETA = 0.9
ENCODING = "rate"
LOG_SCALE = True
EVAL_SPIKE_DRAWS = 5

FAR_TARGETS = [0.01, 0.05]
VAL_FAR_TARGET = 0.01
FAMILY_FAR = 0.05

ABLATION_SEEDS = [0, 1, 2]
ABLATION_MAX_EPOCHS = 10
ABLATION_GRID = {
    "time_steps": [5, 10, 25, 50],
    "beta": [0.5, 0.8, 0.9, 0.95],
    "encoding": ["rate", "direct"],
    "log_scale": [True, False],
}

# 45 nm, 32-bit float (Horowitz, ISSCC 2014)
ENERGY_PJ = {"mac": 4.6, "ac": 0.9}
