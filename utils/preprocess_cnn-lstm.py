import os
import librosa
import numpy as np

# Paths
original_audio_dir = "./data/train/original/"
augmented_audio_dir = "./data/train/augmented_signals/"
mel_original_dir = "./data/train/cnn-lstm/original/"
mel_original_augmented_dir = "./data/train/cnn-lstm/original_augmented/"

# Ensure directories exist
os.makedirs(mel_original_dir, exist_ok=True)
os.makedirs(mel_original_augmented_dir, exist_ok=True)

# Parameters for Mel spectrogram
n_mels = 128
n_fft = 2048
hop_length = 512

# Function to generate and save Mel spectrogram
def save_mel_spectrogram(audio_path, output_path):
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Generate Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save as .npy
        np.save(output_path, log_mel_spec)
        print(f"Saved Mel spectrogram: {output_path}")
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

if __name__ == "__main__":
    # Generate Mel spectrograms for Original Signals
    print("Generating Mel spectrograms for original signals...")
    for file_name in os.listdir(original_audio_dir):
        if file_name.endswith(".wav"):
            audio_path = os.path.join(original_audio_dir, file_name)
            output_path = os.path.join(mel_original_dir, file_name.replace(".wav", ".npy"))
            save_mel_spectrogram(audio_path, output_path)

    # Generate Mel spectrograms for Original + Augmented Signals
    print("Generating Mel spectrograms for original + augmented signals...")
    for file_name in os.listdir(original_audio_dir):
        if file_name.endswith(".wav"):
            audio_path = os.path.join(original_audio_dir, file_name)
            output_path = os.path.join(mel_original_augmented_dir, file_name.replace(".wav", ".npy"))
            save_mel_spectrogram(audio_path, output_path)

    for file_name in os.listdir(augmented_audio_dir):
        if file_name.endswith(".wav"):
            audio_path = os.path.join(augmented_audio_dir, file_name)
            output_path = os.path.join(mel_original_augmented_dir, file_name.replace(".wav", ".npy"))
            save_mel_spectrogram(audio_path, output_path)
