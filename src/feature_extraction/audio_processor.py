
import logging
import librosa
import numpy as np

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_preprocess_audio(
    filepath: str,
    target_sr: int,
    duration: int
) -> np.ndarray:
    """
    Loads an audio file, resamples it, converts to mono, and enforces a fixed duration.

    Args:
        filepath (str): The absolute path to the audio file.
        target_sr (int): The target sampling rate (Hz) for resampling.
        duration (int): The target duration (seconds) for the audio clip.

    Returns:
        np.ndarray: A preprocessed mono audio time series with fixed duration.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: For other errors during audio loading or processing.
    """
    try:
        # 1. Load audio: librosa.load automatically handles various formats (wav, mp3, flac, ogg).
        # It also implicitly converts to float32 and normalizes amplitude to [-1, 1].
        # We load with the original sampling rate first to avoid implicit resampling issues
        # and then explicitly resample to ensure control.
        audio, sr = librosa.load(filepath, sr=None, mono=False) # Load with original SR, keep stereo for now

        # 2. Convert to mono: If the audio is stereo, average the channels.
        # Why: For many audio feature extraction tasks, stereo information is not critical
        # and converting to mono simplifies processing and reduces computational load.
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # 3. Resample to a fixed sampling rate:
        # Why: Consistency across the dataset is crucial for machine learning models.
        # Different sampling rates would lead to different feature dimensions or distorted features.
        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")
            sr = target_sr # Update current sampling rate after resampling

        # Calculate the number of samples for the target duration
        target_total_samples = int(target_sr * duration)

        # 4. Enforce fixed-duration clips via trimming/padding:
        # Why: Machine learning models typically require fixed-size inputs. 
        # Trimming ensures all clips are not longer than the maximum desired duration,
        # and padding ensures all clips meet the minimum duration. Zero-padding is a common
        # and simple method to handle shorter clips without introducing artificial noise.
        if len(audio) > target_total_samples:
            # Trim from the beginning if longer than target duration
            audio = audio[:target_total_samples]
            logging.debug(f"Trimmed audio from {filepath} to {duration} seconds.")
        elif len(audio) < target_total_samples:
            # Pad with zeros if shorter than target duration
            padding = target_total_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode="constant")
            logging.debug(f"Padded audio from {filepath} to {duration} seconds.")

        # Ensure the audio clip is exactly the target duration in samples
        assert len(audio) == target_total_samples, \
            f"Audio length mismatch after processing: Expected {target_total_samples}, got {len(audio)}"

        return audio

    except FileNotFoundError:
        logging.error(f"Audio file not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error processing audio file {filepath}: {e}")
        raise

if __name__ == "__main__":
    # Example usage: This block helps in isolated testing of the audio processor.
    # For actual pipeline usage, use it via scripts/extract_features.py.
    # This requires a dummy audio file to exist for a real test.
    # Create a dummy audio file for testing (requires soundfile library for actual write)
    # You can skip this part if you just want to see the function signature and logging setup.
    
    # Example of how you would use it if you had a dummy audio file
    # from pathlib import Path
    # dummy_audio_path = Path("temp_dummy_audio.wav")
    # if not dummy_audio_path.exists():
    #     # This part requires soundfile and is for creating a *real* dummy audio for test
    #     try:
    #         import soundfile as sf
    #         test_sr = settings.SAMPLING_RATE
    #         test_duration_sec = settings.AUDIO_DURATION + 1 # Make it slightly longer
    #         test_audio = np.random.rand(int(test_sr * test_duration_sec)) * 0.5 # Random audio
    #         sf.write(str(dummy_audio_path), test_audio, test_sr)
    #         logging.info(f"Created dummy audio file for testing: {dummy_audio_path}")
    #     except ImportError:
    #         logging.warning("soundfile not installed. Cannot create a real dummy audio file for testing.")
    #         logging.warning("Please install with: pip install soundfile")
    #         dummy_audio_path = None

    # if dummy_audio_path and dummy_audio_path.exists():
    #     try:
    #         logging.info(f"Processing dummy audio: {dummy_audio_path}")
    #         processed_audio = load_and_preprocess_audio(
    #             str(dummy_audio_path),
    #             settings.SAMPLING_RATE,
    #             settings.AUDIO_DURATION
    #         )
    #         logging.info(f"Processed audio shape: {processed_audio.shape}")
    #         expected_samples = int(settings.SAMPLING_RATE * settings.AUDIO_DURATION)
    #         assert processed_audio.shape[0] == expected_samples,
    #             f"Expected {expected_samples} samples, got {processed_audio.shape[0]}"
    #         logging.info("Dummy audio processing successful and shape is correct.")
    #     except Exception as e:
    #         logging.error(f"Error during dummy audio processing: {e}")
    #     finally:
    #         dummy_audio_path.unlink(missing_ok=True) # Clean up
    # else:
    #     logging.warning("Skipping audio processor example as no real dummy audio file could be created or found.")
    logging.info("Audio processor module ready. Run scripts/extract_features.py for full pipeline test.")

