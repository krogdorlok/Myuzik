"""
VGGish Audio Embedding Extractor

This module provides functionality to extract audio embeddings using the
pretrained VGGish model from TensorFlow Hub.

IMPORTANT: VGGish embeddings are extracted directly from the PREPROCESSED WAVEFORM
produced by src/feature_extraction/audio_processor.py. MFCC features from Phase 2 are
NOT used in this VGGish embedding path. The VGGish model is treated as fully frozen.
"""

import logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variable to cache the loaded VGGish model
_vggish_model = None

def load_vggish_model() -> tf.keras.Model:
    """
    Loads the VGGish model from TensorFlow Hub.

    This function caches the loaded model globally to avoid reloading it
    multiple times during batch processing.

    Returns:
        tf.keras.Model: The loaded VGGish model.
    """
    global _vggish_model

    if _vggish_model is None:
        logging.info(f"Loading VGGish model from TensorFlow Hub: {settings.VGGISH_MODEL_URL}")
        try:
            _vggish_model = hub.load(settings.VGGISH_MODEL_URL)
            logging.info("VGGish model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load VGGish model: {e}")
            raise

    return _vggish_model

def extract_embedding(audio_time_series: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts a fixed-length embedding vector from a preprocessed audio waveform
    using the frozen VGGish model.

    The VGGish model expects 16kHz mono audio. It internally processes the audio
    in 0.96-second frames and produces a 128-dimensional embedding per frame.
    We aggregate frame-level embeddings by averaging to produce a single fixed-length
    embedding vector for the entire audio clip.

    Args:
        audio_time_series (np.ndarray): The preprocessed mono audio time series.
                                        Should be at VGGISH_SAMPLE_RATE (16000 Hz).
        sr (int): The sampling rate of the audio time series. Must match VGGISH_SAMPLE_RATE.

    Returns:
        np.ndarray: A fixed-length embedding vector of shape (128,).

    Raises:
        ValueError: If the sampling rate does not match VGGISH_SAMPLE_RATE.
    """
    if sr != settings.VGGISH_SAMPLE_RATE:
        raise ValueError(
            f"VGGish expects audio at {settings.VGGISH_SAMPLE_RATE} Hz, "
            f"but got {sr} Hz. Please resample audio to {settings.VGGISH_SAMPLE_RATE} Hz."
        )

    if len(audio_time_series.shape) != 1:
        raise ValueError(
            f"Audio time series must be 1D (mono), got shape {audio_time_series.shape}."
        )

    # Load VGGish model (cached)
    model = load_vggish_model()

    try:
        # VGGish expects a float32 tensor with shape [num_samples]
        # The model internally handles framing and spectrogram generation
        audio_tensor = tf.convert_to_tensor(audio_time_series, dtype=tf.float32)

        # Ensure tensor is 1D: [samples]
        if len(audio_tensor.shape) != 1:
            raise ValueError(f"Audio tensor must be 1D, got shape {audio_tensor.shape}")

        # Defensive logging before model inference
        logging.debug(f"Input waveform shape: {audio_tensor.shape}, dtype: {audio_tensor.dtype}")

        # Extract embeddings
        # VGGish expects a 1D tensor and returns embeddings of shape [num_frames, 128]
        embeddings = model(audio_tensor)

        # Defensive logging after model inference
        logging.debug(f"Output embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

        # Average across frames to get a single 128-dimensional vector: [128]
        # This provides a fixed-length representation for variable-length audio clips
        embedding_vector = tf.reduce_mean(embeddings, axis=0)

        # Convert to numpy array
        embedding_array = embedding_vector.numpy()

        logging.debug(f"Extracted VGGish embedding of shape: {embedding_array.shape}")

        return embedding_array

    except Exception as e:
        logging.error(f"Error extracting VGGish embedding: {e}")
        raise

if __name__ == "__main__":
    # Example usage: Helps in isolated testing of the VGGish extractor.

    # Create dummy audio data for testing (3 seconds at 16kHz)
    test_sr = settings.VGGISH_SAMPLE_RATE
    test_duration = settings.AUDIO_DURATION
    dummy_audio = np.random.rand(int(test_sr * test_duration)).astype(np.float32) * 0.1

    logging.info(f"Testing VGGish extractor with dummy audio (shape: {dummy_audio.shape}, sr: {test_sr})")

    try:
        extracted_embedding = extract_embedding(dummy_audio, test_sr)

        if extracted_embedding.shape == (128,):
            logging.info(f"Successfully extracted VGGish embedding. Shape: {extracted_embedding.shape}")
            logging.info("VGGish extractor module ready.")
        else:
            logging.error(f"Unexpected embedding shape: {extracted_embedding.shape}. Expected (128,).")

    except Exception as e:
        logging.error(f"VGGish extractor test failed: {e}")