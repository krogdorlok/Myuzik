"""
Audio Embedding Cache

This module handles caching of extracted audio embeddings to disk.

The cache key depends ONLY on:
- relative audio path (to ensure cross-environment stability)
- embedding model name (e.g., "vggish")
- embedding-relevant parameters (sampling_rate, frame_seconds)
"""

import hashlib
import json
import logging
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _generate_embedding_cache_key(
    audio_filepath_relative: str,
    embedding_model_name: str,
    embedding_params: Dict[str, Any]
) -> str:
    """
    Generates a deterministic hash key for caching based on the audio file path
    and relevant embedding extraction configuration parameters.

    The hash ensures that if any critical parameter changes, a new cache file is
    generated, preventing the use of stale embeddings.

    Args:
        audio_filepath_relative (str): The RELATIVE path to the original audio file.
                                       Using relative paths ensures cache stability across
                                       different environments and absolute paths.
        embedding_model_name (str): The name of the embedding model (e.g., "vggish").
        embedding_params (dict): A dictionary of configuration parameters relevant to
                                 embedding extraction. Only embedding-relevant parameters
                                 are included in the hash.

    Returns:
        str: A SHA256 hash string representing the unique cache key.
    """
    # Filter embedding_params to include ONLY embedding-relevant parameters
    relevant_keys = {"sampling_rate", "frame_seconds"}
    filtered_config = {k: v for k, v in embedding_params.items() if k in relevant_keys}

    # Use a sorted JSON dump for consistent hashing
    config_string = json.dumps(filtered_config, sort_keys=True)
    combined_string = f"{audio_filepath_relative}-{embedding_model_name}-{config_string}"
    return hashlib.sha256(combined_string.encode("utf-8")).hexdigest()

def get_embedding_path(
    audio_filepath: str,
    embedding_model_name: str,
    embedding_params: Dict[str, Any]
) -> Path:
    """
    Determines the full path for a cached embedding file.

    Args:
        audio_filepath (str): The path to the original audio file (absolute or relative).
                              It will be converted to relative for hashing.
        embedding_model_name (str): The name of the embedding model (e.g., "vggish").
        embedding_params (dict): Configuration parameters used for embedding extraction.

    Returns:
        Path: The absolute path where the embedding should be cached.
    """
    # Convert to relative path for consistent caching
    try:
        audio_filepath_relative = os.path.relpath(audio_filepath, settings.BASE_DIR)
    except ValueError:
        # Fallback to absolute if they are on different drives (Windows)
        audio_filepath_relative = audio_filepath

    cache_key = _generate_embedding_cache_key(
        audio_filepath_relative,
        embedding_model_name,
        embedding_params
    )

    # Use the cache key to create a unique filename for the embedding
    filename = f"{cache_key}_{embedding_model_name}.npy"
    return Path(settings.EMBEDDING_CACHE_DIR) / filename

def load_embedding(embedding_path: Path) -> np.ndarray | None:
    """
    Loads a cached embedding from disk.

    Args:
        embedding_path (Path): The path to the cached .npy file.

    Returns:
        np.ndarray | None: The loaded embedding vector, or None if the file does not
                          exist or an error occurs.
    """
    if embedding_path.exists():
        try:
            embedding = np.load(embedding_path)
            logging.debug(f"Loaded cached embedding from: {embedding_path}")
            return embedding
        except Exception as e:
            logging.warning(f"Error loading cached embedding from {embedding_path}: {e}. Recomputing.")
            return None
    logging.debug(f"Cached embedding not found: {embedding_path}")
    return None

def save_embedding(embedding: np.ndarray, embedding_path: Path):
    """
    Saves an embedding array to disk as a .npy file.

    Args:
        embedding (np.ndarray): The embedding vector to save.
        embedding_path (Path): The path where the embedding will be saved.
    """
    os.makedirs(embedding_path.parent, exist_ok=True) # Ensure directory exists
    try:
        np.save(embedding_path, embedding)
        logging.debug(f"Saved embedding to cache: {embedding_path}")
    except Exception as e:
        logging.error(f"Error saving embedding to {embedding_path}: {e}")

if __name__ == "__main__":
    # Example usage: Helps in isolated testing of the embedding caching mechanism.

    # 1. Define dummy inputs
    dummy_audio_filepath = "data/raw/rock/rock1.wav"
    dummy_embedding_model_name = "vggish"
    dummy_embedding_params = {
        "sampling_rate": settings.VGGISH_SAMPLE_RATE,
        "frame_seconds": settings.VGGISH_FRAME_SECONDS
    }
    dummy_embedding_data = np.random.rand(128) # Example VGGish embedding shape

    logging.info("Testing embedding caching module...")

    # 2. Get embedding path
    cache_path = get_embedding_path(
        dummy_audio_filepath,
        dummy_embedding_model_name,
        dummy_embedding_params
    )
    logging.info(f"Generated cache path: {cache_path}")

    # 3. Test saving embedding
    logging.info(f"Attempting to save dummy embedding to {cache_path}...")
    save_embedding(dummy_embedding_data, cache_path)

    # 4. Test loading embedding
    logging.info(f"Attempting to load dummy embedding from {cache_path}...")
    loaded_embedding = load_embedding(cache_path)

    if loaded_embedding is not None:
        logging.info(f"Successfully loaded cached embedding. Shape: {loaded_embedding.shape}")
        assert np.array_equal(dummy_embedding_data, loaded_embedding)
        logging.info("Loaded embedding matches original data.")
    else:
        logging.error("Failed to load cached embedding.")

    # 5. Test cache invalidation by changing params
    logging.info("Testing cache invalidation with changed params...")
    changed_embedding_params = dummy_embedding_params.copy()
    changed_embedding_params["sampling_rate"] = 22050 # Change a parameter
    new_cache_path = get_embedding_path(
        dummy_audio_filepath,
        dummy_embedding_model_name,
        changed_embedding_params
    )
    logging.info(f"Generated new cache path with changed params: {new_cache_path}")
    loaded_with_new_params = load_embedding(new_cache_path)
    if loaded_with_new_params is None:
        logging.info("As expected, cached embedding not found for new params.")
    else:
        logging.error("Cache invalidation failed: Loaded embedding with changed params unexpectedly.")

    # 6. Clean up created dummy cache file
    if cache_path.exists():
        cache_path.unlink()
        logging.info(f"Cleaned up dummy cache file: {cache_path}")
    if new_cache_path.exists():
        new_cache_path.unlink()
        logging.info(f"Cleaned up dummy cache file: {new_cache_path}")

    logging.info("Embedding cache module testing complete.")