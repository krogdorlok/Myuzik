
import json
import logging
from pathlib import Path
import os

from config import settings
from src.feature_extraction.audio_processor import load_and_preprocess_audio
from src.feature_extraction.feature_extractor import extract_features
from src.feature_extraction.feature_cache import get_feature_path, load_feature, save_feature

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_features_pipeline():
    """
    Orchestrates the feature extraction and caching process.
    1. Loads the dataset index.
    2. Iterates through each audio file.
    3. Preprocesses audio, extracts features, and caches them.
    4. Skips re-computation if features are already cached.
    """
    logging.info("Starting feature extraction pipeline...")

    # 1. Load the dataset index
    dataset_index_path = Path(settings.PROCESSED_ROOT) / settings.DATASET_INDEX_FILE
    if not dataset_index_path.exists():
        logging.error(f"Dataset index file not found: {dataset_index_path}. Please run build_dataset_index.py first.")
        return

    with open(dataset_index_path, "r") as f:
        indexed_dataset_info = json.load(f)

    files_to_process = indexed_dataset_info["files"]
    if not files_to_process:
        logging.warning("No audio files found in the dataset index to process.")
        return

    feature_config = {
        "sampling_rate": settings.SAMPLING_RATE,
        "duration": settings.AUDIO_DURATION,
        "n_mfcc": settings.N_MFCC,
        "n_fft": settings.N_FFT,
        "hop_length": settings.HOP_LENGTH
    }

    total_files = len(files_to_process)
    processed_count = 0
    cached_count = 0
    failed_count = 0

    logging.info(f"Processing {total_files} audio files for feature extraction...")

    for i, file_info in enumerate(files_to_process):
        filepath_relative = file_info["filepath"]
        filepath_absolute = Path(settings.BASE_DIR) / filepath_relative
        genre_name = file_info["genre_name"]
        label = file_info["label"]

        logging.debug(f"({i+1}/{total_files}) Processing: {filepath_relative}")

        # Determine cache path for MFCCs
        mfcc_cache_path = get_feature_path(str(filepath_absolute), "mfcc", feature_config)

        # 4. Check if features are already cached
        mfccs = load_feature(mfcc_cache_path)

        if mfccs is not None:
            cached_count += 1
            logging.debug(f"Features for {filepath_relative} already cached. Skipping re-computation.")
            processed_count += 1 # Count as processed even if cached
            continue

        try:
            # 3. Preprocess audio
            audio_time_series = load_and_preprocess_audio(
                str(filepath_absolute),
                settings.SAMPLING_RATE,
                settings.AUDIO_DURATION
            )

            # 3. Extract features
            extracted_features = extract_features(
                audio_time_series,
                settings.SAMPLING_RATE,
                n_mfcc=settings.N_MFCC,
                n_fft=settings.N_FFT,
                hop_length=settings.HOP_LENGTH
            )

            # Cache extracted MFCCs
            if "mfcc" in extracted_features and extracted_features["mfcc"].size > 0:
                save_feature(extracted_features["mfcc"], mfcc_cache_path)
                logging.debug(f"Successfully extracted and cached MFCCs for {filepath_relative}")
                processed_count += 1
            else:
                logging.warning(f"No MFCCs extracted for {filepath_relative}. Skipping cache.")
                failed_count += 1

        except FileNotFoundError:
            logging.error(f"Original audio file not found for processing: {filepath_absolute}")
            failed_count += 1
        except Exception as e:
            logging.error(f"Failed to extract features for {filepath_relative}: {e}")
            failed_count += 1

    logging.info("--- Feature extraction pipeline completed ---")
    logging.info(f"Total files: {total_files}")
    logging.info(f"Files processed (newly extracted or cached): {processed_count}")
    logging.info(f"Files loaded from cache: {cached_count}")
    logging.info(f"Files failed: {failed_count}")

if __name__ == "__main__":
    # Example of how to run the pipeline.
    # Ensure `build_dataset_index.py` has been run first to create `dataset_index.json`.
    logging.info("--- Executing extract_features.py as main ---")
    extract_features_pipeline()
    logging.info("--- extract_features.py execution finished ---")
