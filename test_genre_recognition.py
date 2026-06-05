"""
Simple Genre Recognition Test Script
This script demonstrates how to test the genre recognition system.
"""

import os
import sys
from pathlib import Path


def test_genre_recognition(audio_path):
    """
    Test genre recognition with an audio file.
    """
    print("🎵 Testing Genre Recognition System")
    print(f"📁 Audio file: {audio_path}")
    print("")

    # Check if file exists
    if not Path(audio_path).exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
        return

    # Check disk space
    import shutil

    disk_usage = shutil.disk_usage("/")
    free_space_gb = disk_usage.free / (1024**3)

    print(f"💾 Available disk space: {free_space_gb:.2f} GB")

    if free_space_gb < 0.5:  # Need at least 500MB for VGGish model
        print("❌ Insufficient disk space! Need at least 500MB free.")
        print("Please free up disk space and try again.")
        return

    # Set TensorFlow Hub cache to local directory
    os.environ["TFHUB_CACHE_DIR"] = "./data/tfhub_cache"

    try:
        # Import after setting environment variable
        sys.path.insert(0, ".")
        from scripts.predict_genre import predict_genre

        print("🚀 Starting genre prediction...")
        print("=" * 60)

        # Run prediction
        predict_genre(audio_path)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print("Please check that all dependencies are installed.")


def main():
    """
    Main function to test genre recognition.
    """
    print("🎸 Music Genre Recognition Test")
    print("=" * 60)

    # Available test files
    test_files = [
        "./data/raw/rock/rock1.wav",
        "./data/raw/jazz/jazz1.mp3",
        "./data/raw/rock/rock_song1.wav",
        "./data/raw/jazz/jazz_song1.wav",
    ]

    print("\n📋 Available test files:")
    for i, file in enumerate(test_files, 1):
        exists = "✅" if Path(file).exists() else "❌"
        print(f"{i}. {exists} {file}")

    print("\n💡 Usage:")
    print("python test_genre_recognition.py <path_to_audio_file>")
    print("Or simply run this script to test with rock1.wav")

    # Test with first available file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = test_files[0] if Path(test_files[0]).exists() else test_files[2]

    print(f"\n🎯 Testing with: {audio_file}")
    print("=" * 60)

    test_genre_recognition(audio_file)


if __name__ == "__main__":
    main()
