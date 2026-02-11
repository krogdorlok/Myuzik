
import json
import logging
from pathlib import Path
import os

from config import settings
from src.data_utils.dataset_loader import load_dataset
from src.data_utils.dataset_splitter import create_stratified_split, save_splits

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_dataset_pipeline():
    """
    Orchestrates the dataset indexing and splitting process.
    1. Loads the raw dataset and creates an index.
    2. Creates stratified train/validation splits.
    3. Saves the dataset index and splits to JSON files.
    """
    logging.info("Starting dataset indexing and splitting pipeline...")

    # 1. Load the raw dataset and create an index
    logging.info(f"Loading dataset from raw data directory: {settings.RAW_DATA_DIR}")
    indexed_dataset_info = load_dataset(settings.RAW_DATA_DIR)

    if not indexed_dataset_info["files"]:
        logging.error("No files found in the dataset. Exiting build_dataset_pipeline.")
        return

    # Prepare data for splitting (only the file entries)
    dataset_for_splitting = indexed_dataset_info["files"]

    # 2. Create stratified train/validation splits
    logging.info(f"Creating stratified train/validation splits (ratio: {settings.TRAIN_VAL_SPLIT})...")
    splits = create_stratified_split(
        dataset_for_splitting,
        settings.TRAIN_VAL_SPLIT,
        settings.RANDOM_SEED
    )

    # 3. Save the dataset index and splits to JSON files
    # Save full dataset index (including genre mapping and all file entries)
    dataset_index_output_path = Path(settings.PROCESSED_ROOT) / settings.DATASET_INDEX_FILE
    os.makedirs(Path(settings.PROCESSED_ROOT), exist_ok=True)
    with open(dataset_index_output_path, "w") as f:
        json.dump(indexed_dataset_info, f, indent=4)
    logging.info(f"Dataset index saved to: {dataset_index_output_path}")

    # Save train/validation splits
    splits_output_path = Path(settings.PROCESSED_ROOT) / settings.SPLITS_FILE
    save_splits(splits, splits_output_path)

    logging.info("Dataset indexing and splitting pipeline completed.")

if __name__ == "__main__":
    # Example of how to run the pipeline
    # To test this, you need to have some dummy audio files in data/raw/<genre>/
    logging.info("--- Executing build_dataset_index.py as main ---")
    
    # Create dummy raw data for testing if it doesn't exist
    dummy_rock_dir = os.path.join(settings.RAW_DATA_DIR, "rock")
    dummy_jazz_dir = os.path.join(settings.RAW_DATA_DIR, "jazz")
    os.makedirs(dummy_rock_dir, exist_ok=True)
    os.makedirs(dummy_jazz_dir, exist_ok=True)

    # Create placeholder files. For actual testing, replace with valid audio files.
    # These files are just for ensuring the loader and splitter run without FileNotFoundError.
    # NOTE: librosa.load will still fail on these empty files. You must provide real audio.
    if not Path(os.path.join(dummy_rock_dir, "rock_song1.wav")).exists():
        Path(os.path.join(dummy_rock_dir, "rock_song1.wav")).touch()
    if not Path(os.path.join(dummy_rock_dir, "rock_song2.wav")).exists():
        Path(os.path.join(dummy_rock_dir, "rock_song2.wav")).touch()
    if not Path(os.path.join(dummy_jazz_dir, "jazz_song1.wav")).exists():
        Path(os.path.join(dummy_jazz_dir, "jazz_song1.wav")).touch()

    build_dataset_pipeline()
    logging.info("--- build_dataset_index.py execution finished ---")
