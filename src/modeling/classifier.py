"""
Lightweight Supervised Classifier

This module implements a simple supervised classifier for music genre classification.
Currently, only Logistic Regression is implemented. Other classifiers are listed as TODOs.
"""

import logging
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_classifier(classifier_type: str = settings.CLASSIFIER_TYPE, **kwargs):
    """
    Creates a classifier instance based on the specified type.

    Currently supports 'logistic_regression'. Other classifiers are future work.

    Args:
        classifier_type (str): The type of classifier to create.
        **kwargs: Additional keyword arguments to pass to the classifier constructor.

    Returns:
        The initialized classifier instance.

    Raises:
        ValueError: If an unsupported classifier type is specified.
    """
    classifier_type = classifier_type.lower()

    if classifier_type == "logistic_regression":
        logging.info(f"Creating Logistic Regression classifier with params: {kwargs}")
        # Use multinomial logistic regression for multi-class classification
        # solver='lbfgs' works well for multinomial
        classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', **kwargs)
        return classifier

    # TODO: Add support for other classifiers
    # elif classifier_type == "svm":
    #     from sklearn.svm import LinearSVC
    #     classifier = LinearSVC(**kwargs)
    #     return classifier
    # elif classifier_type == "mlp":
    #     from sklearn.neural_network import MLPClassifier
    #     classifier = MLPClassifier(**kwargs)
    #     return classifier

    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

def train_classifier(classifier, X_train, y_train):
    """
    Trains the classifier on the provided training data.

    Args:
        classifier: The classifier instance to train.
        X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features).
        y_train (np.ndarray): Training labels of shape (n_samples,).

    Returns:
        The trained classifier instance.
    """
    logging.info(f"Training {classifier.__class__.__name__} on {X_train.shape[0]} samples...")

    try:
        classifier.fit(X_train, y_train)
        logging.info(f"Classifier training completed.")
        return classifier
    except Exception as e:
        logging.error(f"Error during classifier training: {e}")
        raise

def save_classifier(classifier, label_mapping: dict, model_path: str = settings.CLASSIFIER_MODEL_PATH):
    """
    Saves a trained classifier and label mapping to disk using joblib.

    Args:
        classifier: The trained classifier instance to save.
        label_mapping (dict): The genre-to-label mapping dictionary.
        model_path (str): The file path where the model will be saved.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save classifier and label mapping as a tuple
        model_data = {
            'classifier': classifier,
            'label_mapping': label_mapping
        }
        joblib.dump(model_data, model_path)
        logging.info(f"Classifier saved to: {model_path}")
    except Exception as e:
        logging.error(f"Error saving classifier to {model_path}: {e}")
        raise

def load_classifier(model_path: str = settings.CLASSIFIER_MODEL_PATH):
    """
    Loads a trained classifier and label mapping from disk.

    Args:
        model_path (str): The file path to load the model from.

    Returns:
        tuple: (classifier, label_mapping) where classifier is the loaded
                classifier instance and label_mapping is the genre-to-label dictionary.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Classifier model not found at: {model_path}")

    try:
        model_data = joblib.load(model_path)
        classifier = model_data['classifier']
        label_mapping = model_data['label_mapping']
        logging.info(f"Classifier loaded from: {model_path}")
        return classifier, label_mapping
    except Exception as e:
        logging.error(f"Error loading classifier from {model_path}: {e}")
        raise

if __name__ == "__main__":
    # Example usage: Helps in isolated testing of the classifier module.

    import numpy as np

    logging.info("Testing classifier module...")

    # Create dummy training data
    n_samples = 100
    n_features = 128  # VGGish embedding size
    n_classes = 3     # e.g., rock, jazz, classical

    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, size=n_samples)

    # Create classifier
    classifier = create_classifier(classifier_type="logistic_regression")

    # Train classifier
    trained_classifier = train_classifier(classifier, X_train, y_train)

    # Save classifier
    save_classifier(trained_classifier)

    # Load classifier
    loaded_classifier = load_classifier()

    # Test prediction
    X_test = np.random.rand(5, n_features)
    predictions = loaded_classifier.predict(X_test)
    logging.info(f"Test predictions: {predictions}")

    # Clean up
    if Path(settings.CLASSIFIER_MODEL_PATH).exists():
        Path(settings.CLASSIFIER_MODEL_PATH).unlink()
        logging.info("Cleaned up test classifier model.")

    logging.info("Classifier module testing complete.")