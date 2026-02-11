
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import StratifiedShuffleSplit

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_stratified_split(
    indexed_data: List[Dict[str, Any]],
    train_ratio: float,
    random_seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Performs a stratified train/validation split on the indexed dataset.

    Args:
        indexed_data (list): A list of dictionaries, each representing an audio file
                             with at least 'filepath' and 'label'.
        train_ratio (float): The proportion of the dataset to include in the train split
                             (e.g., 0.8 for 80% train, 20% validation).
        random_seed (int): Seed for reproducible random operations.

    Returns:
        dict: A dictionary containing 'train' and 'validation' lists of file dictionaries.
    """
    if not indexed_data:
        logging.warning("Indexed data is empty. Cannot create splits.")
        return {"train": [], "validation": []}

    filepaths = [item["filepath"] for item in indexed_data]
    labels = [item["label"] for item in indexed_data]

    # Use StratifiedShuffleSplit to ensure class distribution is preserved
    # We need to reshape labels to a 2D array if they are not already
    # This ensures consistency for sklearn's API
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_seed)

    train_indices, val_indices = next(splitter.split(filepaths, labels))

    train_split = [indexed_data[i] for i in train_indices]
    val_split = [indexed_data[i] for i in val_indices]

    logging.info(f"Dataset split into: Train ({len(train_split)} samples), Validation ({len(val_split)} samples)")

    return {"train": train_split, "validation": val_split}

def save_splits(splits: Dict[str, List[Dict[str, Any]]], output_path: str):
    """
    Saves the train and validation splits to a JSON file.

    Args:
        splits (dict): A dictionary containing 'train' and 'validation' lists of file dictionaries.
        output_path (str): The absolute path where the splits JSON file will be saved.
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=4)
    logging.info(f"Splits saved to: {output_path}")

if __name__ == "__main__":
    # Example usage: This block will not be executed in the final pipeline but is useful for testing
    # Create dummy indexed data for testing
    dummy_indexed_data = [
        {"filepath": "data/raw/rock/rock_1.wav", "genre_name": "rock", "label": 0},
        {"filepath": "data/raw/rock/rock_2.wav", "genre_name": "rock", "label": 0},
        {"filepath": "data/raw/rock/rock_3.wav", "genre_name": "rock", "label": 0},
        {"filepath": "data/raw/jazz/jazz_1.wav", "genre_name": "jazz", "label": 1},
        {"filepath": "data/raw/jazz/jazz_2.wav", "genre_name": "jazz", "label": 1},
        {"filepath": "data/raw/classical/classical_1.wav", "genre_name": "classical", "label": 2},
    ]

    logging.info("Creating stratified splits...")
    splits = create_stratified_split(dummy_indexed_data, settings.TRAIN_VAL_SPLIT, settings.RANDOM_SEED)

    output_path = Path(settings.PROCESSED_ROOT) / settings.SPLITS_FILE
    save_splits(splits, output_path)
