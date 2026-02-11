
import os
import json
import logging
from pathlib import Path
import librosa

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(data_root_dir: str) -> dict:
    """
    Discovers genres from subfolders, validates audio files, and creates an indexed dataset.

    Args:
        data_root_dir (str): The root directory where raw genre subfolders are located (e.g., 'data/raw').

    Returns:
        dict: A dictionary containing:
            - 'label_mapping' (dict): A deterministic mapping of genre names to integer labels.
            - 'files' (list): A list of dictionaries, each representing an audio file with
                              'filepath', 'genre_name', and 'label'.
    """
    indexed_dataset = {
        "label_mapping": {},
        "files": []
    }
    genre_names = sorted([d.name for d in Path(data_root_dir).iterdir() if d.is_dir()])

    if not genre_names:
        logging.warning(f"No genre subfolders found in {data_root_dir}. Please organize your dataset as data/raw/<genre>/*.wav.")
        return indexed_dataset

    # Create a deterministic genre to label mapping
    for i, genre_name in enumerate(genre_names):
        indexed_dataset["label_mapping"][genre_name] = i
    logging.info(f"Discovered genres and their labels: {indexed_dataset['label_mapping']}")

    # Discover and validate audio files
    for genre_name, label in indexed_dataset["label_mapping"].items():
        genre_path = Path(data_root_dir) / genre_name
        valid_files_count = 0
        logging.info(f"Processing genre: {genre_name}")

        for ext in settings.SUPPORTED_AUDIO_FORMATS:
            for audio_file_path in genre_path.glob(f"*{ext}"):
                try:
                    # Attempt to load a small part of the audio to validate it
                    # This is robust to corrupt files, skipping them if they can't be loaded
                    librosa.load(audio_file_path, sr=None, mono=True, duration=0.1, res_type='kaiser_fast')
                    indexed_dataset["files"].append({
                        "filepath": str(audio_file_path.relative_to(settings.BASE_DIR)), # Store relative path
                        "genre_name": genre_name,
                        "label": label
                    })
                    valid_files_count += 1
                except Exception as e:
                    logging.warning(f"Skipping corrupt or unreadable file: {audio_file_path} - Error: {e}")

        if valid_files_count == 0:
            logging.warning(f"No valid audio files found for genre '{genre_name}' in {genre_path}.")
        else:
            logging.info(f"Found {valid_files_count} valid files for genre '{genre_name}'.")

    if not indexed_dataset["files"]:
        logging.error("No audio files were indexed. Please check data/raw directory and file formats.")

    return indexed_dataset

if __name__ == "__main__":
    # Example usage: This block will not be executed in the final pipeline but is useful for testing
    # Create dummy data for testing
    dummy_raw_data_dir = os.path.join(settings.BASE_DIR, "data", "raw")
    os.makedirs(os.path.join(dummy_raw_data_dir, "rock"), exist_ok=True)
    os.makedirs(os.path.join(dummy_raw_data_dir, "jazz"), exist_ok=True)
    # Create dummy files - actual audio files are not created, just placeholders
    Path(os.path.join(dummy_raw_data_dir, "rock", "rock_song1.wav")).touch()
    Path(os.path.join(dummy_raw_data_dir, "jazz", "jazz_song1.mp3")).touch()
    Path(os.path.join(dummy_raw_data_dir, "rock", "rock_song2.flac")).touch()

    logging.info(f"Scanning for dataset in: {settings.RAW_DATA_DIR}")
    dataset = load_dataset(settings.RAW_DATA_DIR)

    output_path = Path(settings.PROCESSED_ROOT) / settings.DATASET_INDEX_FILE
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    logging.info(f"Indexed dataset saved to: {output_path}")

    # Clean up dummy data (optional)
    # import shutil
    # shutil.rmtree(dummy_raw_data_dir)
