
import logging
import librosa
import numpy as np
from typing import Dict, Any

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_features(
    audio_time_series: np.ndarray,
    sr: int,
    n_mfcc: int = settings.N_MFCC,
    n_fft: int = settings.N_FFT,
    hop_length: int = settings.HOP_LENGTH
) -> Dict[str, np.ndarray]:
    """
    Extracts primary audio features (MFCCs) from a preprocessed audio time series.

    Args:
        audio_time_series (np.ndarray): The preprocessed mono audio time series.
        sr (int): The sampling rate of the audio time series.
        n_mfcc (int): Number of MFCCs to compute.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for FFT.

    Returns:
        dict: A dictionary where keys are feature names and values are the computed
              features as NumPy arrays. Current output is {'mfcc': np.ndarray}.
              The MFCC array has shape (n_mfcc, time_frames).

    Notes:
        - The output shape is deterministic: (n_features, time_frames).
        - Features are NOT flattened or reshaped to include batch/channel dimensions.
        - Normalization: MFCCs are intentionally left unnormalized at this stage. 
          Global or per-feature normalization (e.g., zero-mean, unit-variance) is 
          deferred to the model training pipeline to ensure that normalization 
          parameters are computed only on the training split and applied consistently.
    """
    features = {}

    # --- Primary Feature: MFCCs ---
    # Why MFCCs: MFCCs are widely used in speech and music information retrieval due to
    # their ability to represent the timbre of sound. They are robust to variations in
    # loudness and perceived similarly to how the human auditory system processes sound.
    try:
        mfccs = librosa.feature.mfcc(
            y=audio_time_series,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        features["mfcc"] = mfccs
        logging.debug(f"Extracted MFCCs with shape: {mfccs.shape}")
    except Exception as e:
        logging.error(f"Error extracting MFCCs: {e}")
        features["mfcc"] = np.array([]) # Return empty array to indicate failure

    # TODO: Implement optional chroma features here.
    # chroma_stft = librosa.feature.chroma_stft(y=audio_time_series, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # features["chroma"] = chroma_stft

    # TODO: Implement optional spectral features (e.g., spectral centroid, rolloff) here.
    # spectral_centroid = librosa.feature.spectral_centroid(y=audio_time_series, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # features["spectral_centroid"] = spectral_centroid

    return features

if __name__ == "__main__":
    # Example usage: This block helps in isolated testing of the feature extractor.
    # For actual pipeline usage, use it via scripts/extract_features.py.
    # Requires dummy audio data.

    # Create dummy audio time series for testing (e.g., 3 seconds of silence/random noise)
    test_sr = settings.SAMPLING_RATE
    test_duration = settings.AUDIO_DURATION
    dummy_audio = np.random.rand(int(test_sr * test_duration)) * 0.1 # Small random noise

    logging.info(f"Extracting features from dummy audio (shape: {dummy_audio.shape}, sr: {test_sr})")
    extracted_features = extract_features(
        audio_time_series=dummy_audio,
        sr=test_sr,
        n_mfcc=settings.N_MFCC,
        n_fft=settings.N_FFT,
        hop_length=settings.HOP_LENGTH
    )

    if "mfcc" in extracted_features and extracted_features["mfcc"].size > 0:
        mfcc_shape = extracted_features["mfcc"].shape
        logging.info(f"Successfully extracted MFCCs. Shape: {mfcc_shape}")
        expected_time_frames = int(np.ceil(test_sr * test_duration / settings.HOP_LENGTH))
        # The number of time frames can vary slightly depending on padding and librosa's internal calculations
        # A more robust check is to ensure the number of MFCCs is correct and time_frames is reasonable.
        assert mfcc_shape[0] == settings.N_MFCC, \
            f"Expected {settings.N_MFCC} MFCCs, got {mfcc_shape[0]}"
        logging.info("Feature extractor module ready. Run scripts/extract_features.py for full pipeline test.")
    else:
        logging.error("MFCC extraction failed in example usage.")
