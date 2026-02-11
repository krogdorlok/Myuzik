"""
Classifier Training Script

This script orchestrates the training and evaluation of the lightweight
Logistic Regression classifier using cached VGGish embeddings.
"""

import json
import logging
import numpy as np
from pathlib import Path

from config import settings
from src.modeling.classifier import create_classifier, train_classifier, save_classifier
from src.modeling.evaluator import evaluate_classifier
from src.embeddings.embedding_cache import get_embedding_path, load_embedding

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_embeddings_for_split(file_info_list: list, embedding_params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads cached embeddings and labels for a given split (train or validation).

    Args:
        file_info_list (list): List of dictionaries containing 'filepath' and 'label'.
        embedding_params (dict): Parameters used for embedding extraction (for cache lookup).

    Returns:
        tuple: (X, y) where X is the feature matrix (n_samples, n_features)
               and y is the label array (n_samples,).
    """
    X = []
    y = []

    for file_info in file_info_list:
        filepath_relative = file_info["filepath"]
        filepath_absolute = Path(settings.BASE_DIR) / filepath_relative
        label = file_info["label"]

        # Determine cache path for embeddings
        embedding_cache_path = get_embedding_path(
            str(filepath_absolute),
            settings.EMBEDDING_MODEL_NAME,
            embedding_params
        )

        # Load cached embedding
        embedding = load_embedding(embedding_cache_path)

        if embedding is not None:
            X.append(embedding)
            y.append(label)
        else:
            logging.warning(
                f"Embedding not found for {filepath_relative}. "
                "Skipping this file. Run scripts/extract_embeddings.py first."
            )

    if len(X) == 0:
        raise ValueError("No embeddings were loaded for this split. "
                         "Please run scripts/extract_embeddings.py first.")

    X_array = np.array(X)
    y_array = np.array(y)

    logging.info(f"Loaded {len(X)} embeddings for split. X shape: {X_array.shape}, y shape: {y_array.shape}")

    return X_array, y_array

def train_classifier_pipeline():
    """
    Orchestrates the classifier training and evaluation process.
    1. Loads dataset splits and label mapping.
    2. Loads cached embeddings for train and validation sets.
    3. Trains the Logistic Regression classifier.
    4. Evaluates the classifier.
    5. Saves the trained model.
    """
    logging.info("Starting classifier training pipeline...")

    # 1. Load dataset splits
    splits_path = Path(settings.PROCESSED_ROOT) / settings.SPLITS_FILE
    if not splits_path.exists():
        logging.error(
            f"Splits file not found: {splits_path}. "
            "Please run scripts/build_dataset_index.py first."
        )
        return

    with open(splits_path, "r") as f:
        splits_data = json.load(f)

    train_files = splits_data["train"]
    val_files = splits_data["validation"]

    # Load label mapping from dataset index
    dataset_index_path = Path(settings.PROCESSED_ROOT) / settings.DATASET_INDEX_FILE
    if not dataset_index_path.exists():
        logging.error(
            f"Dataset index file not found: {dataset_index_path}. "
            "Please run scripts/build_dataset_index.py first."
        )
        return

    with open(dataset_index_path, "r") as f:
        dataset_index = json.load(f)
    label_mapping = dataset_index["label_mapping"]

    logging.info(f"Dataset splits loaded:")
    logging.info(f"  Train: {len(train_files)} files")
    logging.info(f"  Validation: {len(val_files)} files")
    logging.info(f"  Genres: {list(label_mapping.keys())}")

    # Prepare embedding parameters for cache lookup
    embedding_params = {
        "sampling_rate": settings.VGGISH_SAMPLE_RATE,
        "frame_seconds": settings.VGGISH_FRAME_SECONDS
    }

    # 2. Load cached embeddings
    logging.info("Loading cached embeddings for training split...")
    X_train, y_train = load_embeddings_for_split(train_files, embedding_params)

    logging.info("Loading cached embeddings for validation split...")
    X_val, y_val = load_embeddings_for_split(val_files, embedding_params)

    # 3. Create and train classifier
    logging.info(f"Creating {settings.CLASSIFIER_TYPE} classifier...")
    classifier = create_classifier(classifier_type=settings.CLASSIFIER_TYPE)

    logging.info("Training classifier...")
    trained_classifier = train_classifier(classifier, X_train, y_train)

    # 4. Evaluate classifier
    logging.info("Evaluating classifier on validation set...")
    metrics = evaluate_classifier(trained_classifier, X_val, y_val, label_mapping)

    # 5. Save trained classifier
    logging.info(f"Saving trained classifier to {settings.CLASSIFIER_MODEL_PATH}...")
    save_classifier(trained_classifier, label_mapping)

    logging.info("--- Classifier training pipeline completed ---")
    logging.info(f"Final Validation Accuracy: {metrics['overall_accuracy']:.4f}")

if __name__ == "__main__":
    # Example of how to run the pipeline.
    # Ensure `scripts/build_dataset_index.py` and `scripts/extract_embeddings.py`
    # have been run first.
    logging.info("--- Executing train_classifier.py as main ---")
    train_classifier_pipeline()
    logging.info("--- train_classifier.py execution finished ---")