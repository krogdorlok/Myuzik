
import os
import json
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _generate_cache_key(audio_filepath: str, config_params: Dict[str, Any]) -> str:
    """
    Generates a deterministic hash key for caching based on the audio file path
    and relevant feature extraction configuration parameters.

    The hash ensures that if any critical parameter changes, a new cache file is generated,
    preventing the use of stale features.

    Args:
        audio_filepath (str): The path to the original audio file.
        config_params (dict): A dictionary of configuration parameters relevant to
                              feature extraction (e.g., sampling_rate, duration, n_mfcc, etc.).

    Returns:
        str: A SHA256 hash string representing the unique cache key.
    """
    # Ensure the audio_filepath is relative to the project BASE_DIR to maintain
    # cache stability across different environments/absolute paths.
    try:
        relative_path = os.path.relpath(audio_filepath, settings.BASE_DIR)
    except ValueError:
        # Fallback to absolute if they are on different drives (Windows)
        relative_path = audio_filepath

    # Filter config_params to include ONLY feature-relevant parameters
    relevant_keys = {"sampling_rate", "duration", "n_mfcc", "n_fft", "hop_length"}
    filtered_config = {k: v for k, v in config_params.items() if k in relevant_keys}

    # Use a sorted JSON dump for consistent hashing
    config_string = json.dumps(filtered_config, sort_keys=True)
    combined_string = f"{relative_path}-{config_string}"
    return hashlib.sha256(combined_string.encode("utf-8")).hexdigest()

def get_feature_path(audio_filepath: str, feature_name: str, config_params: Dict[str, Any]) -> Path:
    """
    Determines the full path for a cached feature file.

    Args:
        audio_filepath (str): The original audio file path (used to generate a base name).
        feature_name (str): The name of the feature (e.g., "mfcc").
        config_params (dict): Configuration parameters used for feature extraction.

    Returns:
        Path: The absolute path where the feature should be cached.
    """
    cache_key = _generate_cache_key(audio_filepath, config_params)
    # Use the cache key to create a unique filename for the feature.
    # The feature name is also included for clarity and potential multiple features per audio.
    filename = f"{cache_key}_{feature_name}.npy"
    return Path(settings.FEATURE_CACHE_DIR) / filename

def load_feature(feature_path: Path) -> np.ndarray | None:
    """
    Loads a cached feature from disk.

    Args:
        feature_path (Path): The path to the cached .npy file.

    Returns:
        np.ndarray | None: The loaded feature array, or None if the file does not exist or an error occurs.
    """
    if feature_path.exists():
        try:
            feature = np.load(feature_path)
            logging.debug(f"Loaded cached feature from: {feature_path}")
            return feature
        except Exception as e:
            logging.warning(f"Error loading cached feature from {feature_path}: {e}. Recomputing.")
            return None
    logging.debug(f"Cached feature not found: {feature_path}")
    return None

def save_feature(feature: np.ndarray, feature_path: Path):
    """
    Saves a feature array to disk as a .npy file.

    Args:
        feature (np.ndarray): The feature array to save.
        feature_path (Path): The path where the feature will be saved.
    """
    os.makedirs(feature_path.parent, exist_ok=True) # Ensure directory exists
    try:
        np.save(feature_path, feature)
        logging.debug(f"Saved feature to cache: {feature_path}")
    except Exception as e:
        logging.error(f"Error saving feature to {feature_path}: {e}")

if __name__ == "__main__":
    # Example usage: Helps in isolated testing of the feature caching mechanism.

    # 1. Define dummy inputs
    dummy_audio_filepath = "/path/to/my/dummy_audio.wav"
    dummy_feature_name = "mfcc"
    dummy_config_params = {
        "sampling_rate": settings.SAMPLING_RATE,
        "duration": settings.AUDIO_DURATION,
        "n_mfcc": settings.N_MFCC,
        "n_fft": settings.N_FFT,
        "hop_length": settings.HOP_LENGTH
    }
    dummy_feature_data = np.random.rand(settings.N_MFCC, 100) # Example MFCC shape

    logging.info("Testing feature caching module...")

    # 2. Get feature path
    cache_path = get_feature_path(dummy_audio_filepath, dummy_feature_name, dummy_config_params)
    logging.info(f"Generated cache path: {cache_path}")

    # 3. Test saving feature
    logging.info(f"Attempting to save dummy feature to {cache_path}...")
    save_feature(dummy_feature_data, cache_path)

    # 4. Test loading feature
    logging.info(f"Attempting to load dummy feature from {cache_path}...")
    loaded_feature = load_feature(cache_path)

    if loaded_feature is not None:
        logging.info(f"Successfully loaded cached feature. Shape: {loaded_feature.shape}")
        assert np.array_equal(dummy_feature_data, loaded_feature)
        logging.info("Loaded feature matches original data.")
    else:
        logging.error("Failed to load cached feature.")

    # 5. Test cache invalidation by changing config
    logging.info("Testing cache invalidation with changed config...")
    changed_config_params = dummy_config_params.copy()
    changed_config_params["n_mfcc"] = 50 # Change a parameter
    new_cache_path = get_feature_path(dummy_audio_filepath, dummy_feature_name, changed_config_params)
    logging.info(f"Generated new cache path with changed config: {new_cache_path}")
    loaded_with_new_config = load_feature(new_cache_path)
    if loaded_with_new_config is None:
        logging.info("As expected, cached feature not found for new config.")
    else:
        logging.error("Cache invalidation failed: Loaded feature with changed config unexpectedly.")

    # 6. Clean up created dummy cache file
    if cache_path.exists():
        cache_path.unlink()
        logging.info(f"Cleaned up dummy cache file: {cache_path}")
    if new_cache_path.exists():
        new_cache_path.unlink()
        logging.info(f"Cleaned up dummy cache file: {new_cache_path}")

    logging.info("Feature cache module testing complete.")
