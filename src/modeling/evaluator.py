"""
Model Evaluation

This module provides functionality for evaluating the trained classifier,
including overall accuracy, per-genre accuracy, and confusion matrix.
"""

import logging
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_classifier(
    classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_mapping: Dict[str, int]
) -> Dict[str, Any]:
    """
    Evaluates the trained classifier on the test set and returns metrics.

    Args:
        classifier: The trained classifier instance.
        X_test (np.ndarray): Test feature matrix of shape (n_samples, n_features).
        y_test (np.ndarray): Test labels of shape (n_samples,).
        label_mapping (dict): Mapping of genre names to integer labels from
                             dataset_index.json. Used for human-readable reports.

    Returns:
        dict: A dictionary containing evaluation metrics:
            - 'overall_accuracy': Overall accuracy score
            - 'classification_report': Full classification report string
            - 'confusion_matrix': Confusion matrix as numpy array
            - 'confusion_matrix_labeled': Confusion matrix with genre labels (list of lists)
    """
    logging.info(f"Evaluating classifier on {X_test.shape[0]} test samples...")

    # Generate predictions
    y_pred = classifier.predict(X_test)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Generate classification report (per-genre accuracy, precision, recall, F1)
    classification_report_str = classification_report(
        y_test,
        y_pred,
        target_names=label_mapping.keys(),
        zero_division=0
    )
    logging.info("\nClassification Report:\n" + classification_report_str)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create labeled confusion matrix for better readability
    conf_matrix_labeled = _create_labeled_confusion_matrix(
        conf_matrix,
        list(label_mapping.keys())
    )

    logging.info("\nConfusion Matrix:")
    for row in conf_matrix_labeled:
        logging.info(str(row))

    return {
        'overall_accuracy': overall_accuracy,
        'classification_report': classification_report_str,
        'confusion_matrix': conf_matrix,
        'confusion_matrix_labeled': conf_matrix_labeled
    }

def _create_labeled_confusion_matrix(
    conf_matrix: np.ndarray,
    genre_names: list
) -> list:
    """
    Creates a human-readable confusion matrix with genre labels.

    Args:
        conf_matrix (np.ndarray): The confusion matrix from sklearn.
        genre_names (list): List of genre names (ordered same as labels).

    Returns:
        list: A list of lists representing the confusion matrix with labels.
              First row/column contains genre names.
    """
    n_classes = len(genre_names)

    # Create header row
    labeled_matrix = [["\\"] + genre_names]

    # Add data rows with row labels
    for i in range(n_classes):
        row = [genre_names[i]] + conf_matrix[i].tolist()
        labeled_matrix.append(row)

    return labeled_matrix

if __name__ == "__main__":
    # Example usage: Helps in isolated testing of the evaluator module.

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    logging.info("Testing evaluator module...")

    # Create dummy data
    n_samples = 100
    n_features = 128
    n_classes = 3

    X_test = np.random.rand(n_samples, n_features)
    y_test = np.random.randint(0, n_classes, size=n_samples)

    # Create and train a dummy classifier
    classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    classifier.fit(X_test, y_test)

    # Create label mapping
    label_mapping = {"rock": 0, "jazz": 1, "classical": 2}

    # Evaluate
    metrics = evaluate_classifier(classifier, X_test, y_test, label_mapping)

    logging.info(f"\nTest Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    logging.info(f"Confusion Matrix Shape: {metrics['confusion_matrix'].shape}")

    logging.info("Evaluator module testing complete.")