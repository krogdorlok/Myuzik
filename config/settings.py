
import os

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_ROOT = os.path.join(DATA_ROOT, "processed")
FEATURE_CACHE_DIR = os.path.join(PROCESSED_ROOT, "features")

# Ensure output directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_ROOT, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# --- Audio Preprocessing Parameters ---
SAMPLING_RATE = 22050  # Target sampling rate for all audio (Hz)
AUDIO_DURATION = 3     # Fixed duration for audio clips (seconds)

# --- Feature Extraction Parameters (MFCC) ---
N_MFCC = 40            # Number of MFCCs to extract
N_FFT = 2048           # FFT window size
HOP_LENGTH = 512       # Hop length for FFT

# --- Dataset Splitting Parameters ---
TRAIN_VAL_SPLIT = 0.8  # Ratio for training set (e.g., 0.8 for 80% train, 20% validation)
RANDOM_SEED = 42       # Seed for reproducible random operations

# --- File Naming ---
DATASET_INDEX_FILE = "dataset_index.json"
SPLITS_FILE = "splits.json"

# --- Supported Audio Formats ---
SUPPORTED_AUDIO_FORMATS = (".wav", ".mp3", ".flac", ".ogg")

# --- Embedding Extraction Parameters (VGGish) ---
EMBEDDING_CACHE_DIR = os.path.join(PROCESSED_ROOT, "embeddings")
EMBEDDING_MODEL_NAME = "vggish"
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"
VGGISH_SAMPLE_RATE = 16000 # VGGish expects 16kHz audio
VGGISH_FRAME_SECONDS = 0.96 # VGGish processes audio in 0.96-second frames

# Ensure embedding cache directory exists
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# --- Classifier Training Parameters ---
CLASSIFIER_MODEL_PATH = os.path.join(PROCESSED_ROOT, "classifier_model.pkl")
CLASSIFIER_TYPE = "logistic_regression" # Only Logistic Regression for now, others as TODO

# --- Evaluation ---
# No specific parameters for evaluation beyond standard metrics
