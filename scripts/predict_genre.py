"""
Genre Prediction Script

This script takes an audio file, extracts VGGish embeddings, and uses the trained
classifier to predict the music genre.
"""

import argparse
import logging
import numpy as np
from pathlib import Path

from config import settings
from src.feature_extraction.audio_processor import load_and_preprocess_audio
from src.embeddings.vggish_extractor import extract_embedding
from src.modeling.classifier import load_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def predict_genre(audio_filepath: str, classifier_path: str = None):
    """
    Predicts the genre of an audio file using the trained classifier.

    Args:
        audio_filepath (str): Path to the audio file.
        classifier_path (str, optional): Path to the saved classifier model.
                                         If None, uses default from settings.
    """
    if classifier_path is None:
        classifier_path = settings.CLASSIFIER_MODEL_PATH

    # Load the trained classifier
    logging.info(f"Loading classifier from {classifier_path}...")
    classifier, label_mapping = load_classifier(classifier_path)
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # Preprocess audio to VGGish requirements
    logging.info(f"Loading and preprocessing audio: {audio_filepath}")
    try:
        audio_time_series = load_and_preprocess_audio(
            audio_filepath,
            settings.VGGISH_SAMPLE_RATE,
            settings.AUDIO_DURATION
        )
        logging.info(f"Audio loaded successfully. Shape: {audio_time_series.shape}")
    except Exception as e:
        logging.error(f"Failed to load audio file: {e}")
        return

    # Extract VGGish embedding (already averaged across frames)
    logging.info("Extracting VGGish embedding...")
    try:
        embedding = extract_embedding(audio_time_series, settings.VGGISH_SAMPLE_RATE)
        logging.info(f"Embedding extracted. Shape: {embedding.shape}")
    except Exception as e:
        logging.error(f"Failed to extract embedding: {e}")
        return

    # Reshape for prediction (classifier expects 2D: [n_samples, n_features])
    embedding_reshaped = embedding.reshape(1, -1)

    # Predict genre
    logging.info("Predicting genre...")
    try:
        # Get predicted class and probabilities
        predicted_label = classifier.predict(embedding_reshaped)[0]
        probabilities = classifier.predict_proba(embedding_reshaped)[0]

        # Map label to genre name
        predicted_genre = inverse_label_mapping[predicted_label]
        confidence = probabilities[predicted_label]

        # Display results
        print("\n" + "="*60)
        print(f"Predicted Genre: {predicted_genre}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("="*60)

        # Display class probabilities
        print("\nClass Probabilities:")
        sorted_labels = sorted(inverse_label_mapping.items(), key=lambda x: x[0])
        for label, genre_name in sorted_labels:
            prob = probabilities[label]
            print(f"  {genre_name:15s}: {prob*100:.2f}%")
        print("="*60)

        logging.info(f"Prediction complete. Genre: {predicted_genre}, Confidence: {confidence:.4f}")

    except Exception as e:
        logging.error(f"Failed to predict genre: {e}")
        raise

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict music genre from an audio file using the trained classifier."
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to classify."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Path to the saved classifier model (default: {settings.CLASSIFIER_MODEL_PATH})."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logging.error(f"Audio file not found: {args.audio_file}")
        return

    # Run prediction
    predict_genre(str(audio_path), args.model)

if __name__ == "__main__":
    main()