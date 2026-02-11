"""
Embedding Extraction Script

This script orchestrates the extraction and caching of audio embeddings using
the VGGish model. It reuses audio preprocessing from Phase 2.

IMPORTANT: VGGish embeddings are extracted directly from the PREPROCESSED WAVEFORM
produced by src/feature_extraction/audio_processor.py. MFCC features from Phase 2 are
NOT used in this VGGish embedding path.
"""

import json
import logging
from pathlib import Path

from config import settings
from src.feature_extraction.audio_processor import load_and_preprocess_audio
from src.embeddings.vggish_extractor import extract_embedding
from src.embeddings.embedding_cache import get_embedding_path, load_embedding, save_embedding

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_embeddings_pipeline():
    """
    Orchestrates the embedding extraction and caching process.
    1. Loads the dataset index.
    2. Iterates through each audio file.
    3. Preprocesses audio to VGGish requirements.
    4. Extracts VGGish embeddings.
    5. Caches embeddings to disk.
    6. Skips re-computation if embeddings are already cached.
    """
    logging.info("Starting embedding extraction pipeline...")

    # 1. Load the dataset index
    dataset_index_path = Path(settings.PROCESSED_ROOT) / settings.DATASET_INDEX_FILE
    if not dataset_index_path.exists():
        logging.error(
            f"Dataset index file not found: {dataset_index_path}. "
            "Please run scripts/build_dataset_index.py first."
        )
        return

    with open(dataset_index_path, "r") as f:
        indexed_dataset_info = json.load(f)

    files_to_process = indexed_dataset_info["files"]
    if not files_to_process:
        logging.warning("No audio files found in the dataset index to process.")
        return

    # Prepare embedding parameters for cache key generation
    embedding_params = {
        "sampling_rate": settings.VGGISH_SAMPLE_RATE,
        "frame_seconds": settings.VGGISH_FRAME_SECONDS
    }

    total_files = len(files_to_process)
    processed_count = 0
    cached_count = 0
    failed_count = 0

    logging.info(f"Processing {total_files} audio files for VGGish embedding extraction...")

    for i, file_info in enumerate(files_to_process):
        filepath_relative = file_info["filepath"]
        filepath_absolute = Path(settings.BASE_DIR) / filepath_relative
        genre_name = file_info["genre_name"]
        label = file_info["label"]

        logging.debug(f"({i+1}/{total_files}) Processing: {filepath_relative}")

        # Determine cache path for embeddings
        embedding_cache_path = get_embedding_path(
            str(filepath_absolute),
            settings.EMBEDDING_MODEL_NAME,
            embedding_params
        )

        # 6. Check if embeddings are already cached
        embedding = load_embedding(embedding_cache_path)

        if embedding is not None:
            cached_count += 1
            logging.debug(f"Embedding for {filepath_relative} already cached. Skipping re-computation.")
            processed_count += 1  # Count as processed even if cached
            continue

        try:
            # 3. Preprocess audio to VGGish requirements
            # IMPORTANT: We use VGGISH_SAMPLE_RATE (16000 Hz) here, not SAMPLING_RATE (22050 Hz)
            # because VGGish expects 16kHz audio. This is a preprocessing requirement
            # separate from Phase 2's MFCC extraction parameters.
            audio_time_series = load_and_preprocess_audio(
                str(filepath_absolute),
                settings.VGGISH_SAMPLE_RATE,
                settings.AUDIO_DURATION
            )

            # 4. Extract VGGish embeddings
            # The embedding extractor expects audio at VGGISH_SAMPLE_RATE
            extracted_embedding = extract_embedding(
                audio_time_series,
                settings.VGGISH_SAMPLE_RATE
            )

            # 5. Cache extracted embeddings
            if extracted_embedding.size > 0:
                save_embedding(extracted_embedding, embedding_cache_path)
                logging.debug(f"Successfully extracted and cached VGGish embedding for {filepath_relative}")
                processed_count += 1
            else:
                logging.warning(f"No VGGish embedding extracted for {filepath_relative}. Skipping cache.")
                failed_count += 1

        except FileNotFoundError:
            logging.error(f"Original audio file not found for processing: {filepath_absolute}")
            failed_count += 1
        except Exception as e:
            logging.error(f"Failed to extract VGGish embedding for {filepath_relative}: {e}")
            failed_count += 1

    logging.info("--- Embedding extraction pipeline completed ---")
    logging.info(f"Total files: {total_files}")
    logging.info(f"Files processed (newly extracted or cached): {processed_count}")
    logging.info(f"Files loaded from cache: {cached_count}")
    logging.info(f"Files failed: {failed_count}")

if __name__ == "__main__":
    # Example of how to run the pipeline.
    # Ensure `scripts/build_dataset_index.py` has been run first to create `dataset_index.json`.
    logging.info("--- Executing extract_embeddings.py as main ---")
    extract_embeddings_pipeline()
    logging.info("--- extract_embeddings.py execution finished ---")