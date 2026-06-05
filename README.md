# Myuzik - Music Genre Recognition

Instantly classify any song into 10 music genres using AI. Drop in an audio file and get the genre in seconds.

## Features

- **10 Genres**: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock
- **87.8% Accuracy**: Pre-trained model ready to use
- **Easy to Use**: One command to classify any song
- **Multiple Formats**: MP3, WAV, FLAC, OGG, and more

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Classify your song
python test_genre_recognition.py path/to/your/song.mp3
```

That's it! You'll see the predicted genre and confidence level.

## Example Output

```
============================================================
Predicted Genre: rock
Confidence: 84.32%
============================================================

Class Probabilities:
  rock       : 84.32%
  metal      : 8.21%
  blues      : 4.15%
  jazz       : 2.11%
  pop        : 1.22%
  ...
```

## What You Need

- Python 3.8+
- Internet connection (first run only - downloads VGGish model)

## Requirements

See `requirements.txt` for full list. Key dependencies:
- TensorFlow
- Librosa
- Scikit-learn
- NumPy

## FAQ

**Q: What audio formats work?**  
A: MP3, WAV, FLAC, OGG, and most common audio formats.

**Q: How accurate is it?**  
A: ~88% accuracy on validation set. Best on: Blues, Classical, Metal.

**Q: Can I retrain the model?**  
A: This is an inference-only release. Training code is excluded for simplicity.

**Q: What if confidence is low?**  
A: The model might struggle with mixed genres or very short clips. Try a longer sample (30+ seconds).

## License

MIT License - feel free to use in your projects!
