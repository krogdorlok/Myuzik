# Music Genre Differentiation and Recognition from Audio

## Project Goal (High-Level)

This project aims to build an end-to-end machine learning pipeline that classifies music genre purely from audio signals. This involves audio signal processing, feature extraction, and supervised learning for multi-class classification. This project is designed to be production-grade, modular, and extensible.

## Current Scope: Phase 1, Phase 2, & Phase 3B (Pretrained Audio Embeddings & Classification)

This repository currently implements and finalizes the following phases of the project:

### Phase 1: Dataset Handling

This phase focuses on robustly loading, indexing, and splitting audio datasets. Key functionalities include:

- **Automatic Genre Discovery**: Identifies genres from subfolder names within the `data/raw/` directory.
- **Deterministic Label Mapping**: Assigns integer labels to genres in a consistent, reproducible manner.
- **Audio File Validation**: Checks for supported audio formats (`.wav`, `.mp3`, `.flac`, `.ogg`) and validates file integrity, skipping corrupt or unreadable files.
- **Dataset Indexing**: Creates a `dataset_index.json` file containing a mapping of genre names to labels and a list of all valid audio files with their respective metadata.
- **Stratified Dataset Splitting**: Generates `splits.json` for reproducible train/validation splits, ensuring proportional representation of genres.

### Phase 2: Feature Extraction

This phase implements the audio signal processing and feature extraction pipeline. Key functionalities include:

- **Audio Preprocessing**: Loads audio files, resamples them to a fixed sampling rate, converts them to mono, and enforces a consistent audio duration through trimming or zero-padding.
- **MFCC Feature Extraction**: Extracts Mel-frequency cepstral coefficients (MFCCs) as the primary audio feature. The output shape is deterministic `(n_features, time_frames)` and not flattened. MFCCs are intentionally left unnormalized at this stage to allow split-aware normalization during model training.
- **Feature Caching**: Implements a robust caching mechanism to store extracted features on disk (`data/processed/features/`), avoiding redundant computation. Cache keys are generated deterministically based on relative audio file paths and feature-relevant extraction parameters.

### Phase 3B: Pretrained Audio Embeddings & Classification

This phase implements a pretrained audio embedding model and a lightweight supervised classifier for music genre classification. Key functionalities include:

- **VGGish Audio Embeddings**: Extracts 128-dimensional embeddings from audio waveforms using the pretrained VGGish model from TensorFlow Hub. The VGGish model is treated as fully frozen and is not fine-tuned. Embeddings are extracted directly from the preprocessed waveform (resampled to 16kHz mono), not from the Phase 2 MFCC features.
- **Embedding Caching**: Implements a robust caching mechanism similar to Phase 2, but for embeddings. Cache keys depend only on relative audio paths and embedding-relevant parameters (sampling_rate, frame_seconds).
- **Lightweight Classifier**: Implements a Logistic Regression classifier (multi-class, using multinomial logistic regression) trained on the cached VGGish embeddings. The classifier is production-grade, modular, and extensible (other classifiers listed as TODOs).
- **Model Evaluation**: Evaluates the trained classifier on the validation set using standard metrics including overall accuracy, per-genre accuracy (precision, recall, F1-score), and confusion matrix.
- **Model Persistence**: Saves the trained classifier to disk for future use and evaluation.

## Out of Scope (For Now)

The following functionalities are intentionally _not_ implemented in this phase and may be addressed in future phases:

- Additional classifier architectures (e.g., SVMs, Random Forests, MLPs) - listed as TODOs
- Fine-tuning of the VGGish model (it is frozen)
- Real-time Audio Processing or UI
- Advanced model architectures or hyperparameter tuning

## Project Structure

```
music_recog/
├── data/
│   ├── raw/                     # Raw audio files, organized by genre subfolders
│   │   ├── rock/                # e.g., data/raw/rock/song1.wav
│   │   └── jazz/
│   └── processed/
│       ├── dataset_index.json   # JSON file with genre mapping and all indexed file paths
│       ├── splits.json          # JSON file with stratified train/validation splits
│       ├── features/            # Directory for cached MFCC features (e.g., .npy files)
│       └── embeddings/          # Directory for cached VGGish embeddings (e.g., .npy files)
│
├── config/
│   └── settings.py              # Centralized configuration parameters
│
├── src/
│   ├── data_utils/
│   │   ├── dataset_loader.py    # Logic for discovering, validating, and indexing the dataset
│   │   └── dataset_splitter.py  # Logic for creating stratified train/validation splits
│   │
│   ├── feature_extraction/
│   │   ├── audio_processor.py   # Handles audio loading, resampling, mono conversion, trimming/padding
│   │   ├── feature_extractor.py # Computes MFCCs and other specified features
│   │   └── feature_cache.py     # Manages loading and saving of cached MFCC features
│   │
│   ├── embeddings/
│   │   ├── vggish_extractor.py  # Extracts embeddings using the frozen VGGish model
│   │   └── embedding_cache.py   # Manages loading and saving of cached embeddings
│   │
│   └── modeling/
│       ├── classifier.py         # Logistic Regression classifier (extensible for others)
│       └── evaluator.py          # Evaluates classifier using accuracy, classification report, confusion matrix
│
├── scripts/
│   ├── build_dataset_index.py   # Orchestrates dataset indexing and splitting
│   ├── extract_features.py       # Orchestrates MFCC feature extraction and caching
│   ├── extract_embeddings.py     # Orchestrates VGGish embedding extraction and caching
│   └── train_classifier.py       # Orchestrates classifier training and evaluation
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview and instructions
```

## Setup and Usage

### 1. Clone the repository:

```bash
git clone <repository-url>
cd music_recog
```

### 2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Organize your raw dataset:

Place your audio files in the `data/raw/` directory, organized into subfolders by genre.
Example:

```
music_recog/
├── data/
│   ├── raw/
│   │   ├── rock/
│   │   │   └── song1.wav
│   │   │   └── song2.mp3
│   │   ├── jazz/
│   │   │   └── trackA.flac
│   │   └── classical/
```

### 4. Configure `config/settings.py` (Optional):

Review and adjust parameters in `config/settings.py` as needed (e.g., `SAMPLING_RATE`, `AUDIO_DURATION`, `N_MFCC`).

### 5. Build the Dataset Index and Splits:

Run the script to discover audio files, validate them, and create the `dataset_index.json` and `splits.json` files.

```bash
python scripts/build_dataset_index.py
```

This will generate:

- `data/processed/dataset_index.json`
- `data/processed/splits.json`

### 6. Extract and Cache Features:

Run the script to preprocess audio, extract features, and cache them to disk. This process will automatically skip re-computation for already cached features.

```bash
python scripts/extract_features.py
```

This will populate the `data/processed/features/` directory with `.npy` files containing the extracted MFCCs.

### 7. Extract and Cache VGGish Embeddings:

Run the script to preprocess audio (specifically for VGGish's 16kHz requirement), extract VGGish embeddings, and cache them to disk. This process will automatically skip re-computation for already cached embeddings.

```bash
python scripts/extract_embeddings.py
```

This will populate the `data/processed/embeddings/` directory with `.npy` files containing the extracted VGGish embeddings.

**Note:** VGGish embeddings are extracted directly from the preprocessed waveform, not from the Phase 2 MFCC features. The audio is preprocessed specifically for VGGish's requirements (16kHz mono).

### 8. Train and Evaluate Classifier:

Run the script to load the cached embeddings, train the Logistic Regression classifier on the training set, and evaluate it on the validation set.

```bash
python scripts/train_classifier.py
```

This will:

- Load the cached VGGish embeddings for train and validation splits
- Train a Logistic Regression classifier
- Evaluate the classifier using accuracy, classification report (per-genre metrics), and confusion matrix
- Save the trained classifier to `data/processed/classifier_model.pkl`

The evaluation metrics will be logged to the console, providing insights into model performance.

## Next Steps (Future Phases)

With the dataset indexed, split, embeddings extracted, and a classifier trained and evaluated, potential future enhancements could include:

- Exploring additional classifier architectures (e.g., SVMs, Random Forests, MLPs)
- Fine-tuning the VGGish model or exploring other pretrained audio embedding models
- Implementing cross-validation and hyperparameter tuning
- Building real-time inference capabilities
- Developing a user interface
- Deployment considerations and model optimization
