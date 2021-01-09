import os


RAW_DIR = "data_raw"
DATASETS_DIR = "datasets"
TRAINED_MODELS_DIR = "trained_models"
CACHE_DIR = ".cache"

OLD_DATASET = "all_cancer_types_2007"
OLD_HISTOLOGIES_FILE = os.path.join(RAW_DIR, "ISTOLOGIE_corr.csv")
OLD_NEOPLASMS_FILE = os.path.join(RAW_DIR, "RTRT_NEOPLASI_corr.csv")

NEW_DATASET = "breast_and_colon_cancer_2015"
NEW_DATA_FILE = os.path.join(RAW_DIR, "dati.csv")

# GLOVE_FILE = os.path.join(DATA_DIR, "glove.xxx")

IDF = "idf.json"
TOKEN_CODEC = "token_codec.json"
STATS = "statistics"

TRAINING_SET = "training_set.csv"
VALIDATION_SET = "validation_set.csv"
TEST_SET = "test_set.csv"
UNSUPERVISED_SET = "unsupervised_set.csv"
