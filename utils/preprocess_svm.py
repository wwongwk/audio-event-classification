import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from itertools import chain

# Paths
metadata_path = "./data/raw/meta/esc50.csv"
train_audio_dir = "./data/train/original/"
test_audio_dir = "./data/test/original/"
output_train_dir = "./data/train/svm/"
output_test_dir = "./data/test/svm/"

# Ensure output directories exist
methods = ["method_1_pca", "method_2_stats", "method_3_random", "method_4_continuous", "method_5_spaced", "method_6_offset"]
for method in methods:
    os.makedirs(os.path.join(output_train_dir, method), exist_ok=True)
    os.makedirs(os.path.join(output_test_dir, method), exist_ok=True)

# Helper functions
def return_mfcc(audio, sr, n_mfcc=12):
    """Compute 12 MFCCs."""
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, fmax=sr / 2.0)
    return librosa.feature.mfcc(S=librosa.power_to_db(mel_spect), sr=sr, n_mfcc=n_mfcc)

def random_frames(data, n_frames=100):
    """Select random frames."""
    return list(chain.from_iterable(data[:, np.random.choice(data.shape[1], n_frames, replace=False)].T))

def continuous_frames(data, n_frames=100, start=0):
    """Select continuous frames."""
    return list(chain.from_iterable(data[:, start:start + n_frames].T))

def spaced_frames(data, step=15):
    """Pick every nth frame."""
    return list(chain.from_iterable(data[:, ::step].T))

# Preprocessing methods
def preprocess_method_1(X_files, audio_dir, output_dir):
    """Method 1: PCA-ready MFCCs (stored as raw data)."""
    for _, row in tqdm(X_files.iterrows(), total=len(X_files)):
        path = os.path.join(audio_dir, row["filename"])
        y, sr = librosa.load(path, sr=None)
        mfcc = return_mfcc(y, sr)
        data = list(chain.from_iterable(mfcc.T))  # Flatten MFCCs
        np.save(os.path.join(output_dir, row["filename"].replace(".wav", ".npy")), data)

def preprocess_method_2(X_files, audio_dir, output_dir):
    """Method 2: Compact Features."""
    for _, row in tqdm(X_files.iterrows(), total=len(X_files)):
        path = os.path.join(audio_dir, row["filename"])
        y, sr = librosa.load(path, sr=None)
        mfcc = return_mfcc(y, sr)
        energy = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        features = []
        # Mean and stddev for MFCC, energy, and ZCR
        features.extend(np.mean(mfcc, axis=1))
        features.extend([np.mean(energy), np.mean(zcr)])
        features.extend(np.std(mfcc, axis=1))
        features.extend([np.std(energy), np.std(zcr)])
        # Delta and delta-delta
        features.extend(np.mean(librosa.feature.delta(mfcc), axis=1))
        features.extend([np.mean(librosa.feature.delta(energy)), np.mean(librosa.feature.delta(zcr))])
        features.extend(np.mean(librosa.feature.delta(mfcc, order=2), axis=1))
        features.extend([np.mean(librosa.feature.delta(energy, order=2)), np.mean(librosa.feature.delta(zcr, order=2))])
        np.save(os.path.join(output_dir, row["filename"].replace(".wav", ".npy")), features)

def preprocess_frames(X_files, audio_dir, output_dir, method, n_frames=100, step=15, start=0):
    """Generic frame selection method."""
    for _, row in tqdm(X_files.iterrows(), total=len(X_files)):
        path = os.path.join(audio_dir, row["filename"])
        y, sr = librosa.load(path, sr=None)
        mfcc = return_mfcc(y, sr)
        if method == "random":
            data = random_frames(mfcc, n_frames)
        elif method == "continuous":
            data = continuous_frames(mfcc, n_frames, start)
        elif method == "spaced":
            data = spaced_frames(mfcc, step)
        np.save(os.path.join(output_dir, row["filename"].replace(".wav", ".npy")), data)

# Main function
if __name__ == "__main__":
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    train_files = metadata[metadata["filename"].isin(os.listdir(train_audio_dir))]
    test_files = metadata[metadata["filename"].isin(os.listdir(test_audio_dir))]

    # Process each method
    print("Processing Method 1...")
    preprocess_method_1(train_files, train_audio_dir, os.path.join(output_train_dir, "method_1_pca"))
    preprocess_method_1(test_files, test_audio_dir, os.path.join(output_test_dir, "method_1_pca"))

    print("Processing Method 2...")
    preprocess_method_2(train_files, train_audio_dir, os.path.join(output_train_dir, "method_2_stats"))
    preprocess_method_2(test_files, test_audio_dir, os.path.join(output_test_dir, "method_2_stats"))

    print("Processing Method 3...")
    preprocess_frames(train_files, train_audio_dir, os.path.join(output_train_dir, "method_3_random"), "random", n_frames=100)
    preprocess_frames(test_files, test_audio_dir, os.path.join(output_test_dir, "method_3_random"), "random", n_frames=100)

    print("Processing Method 4...")
    preprocess_frames(train_files, train_audio_dir, os.path.join(output_train_dir, "method_4_continuous"), "continuous", n_frames=100, start=0)
    preprocess_frames(test_files, test_audio_dir, os.path.join(output_test_dir, "method_4_continuous"), "continuous", n_frames=100, start=0)

    print("Processing Method 5...")
    preprocess_frames(train_files, train_audio_dir, os.path.join(output_train_dir, "method_5_spaced"), "spaced", step=15)
    preprocess_frames(test_files, test_audio_dir, os.path.join(output_test_dir, "method_5_spaced"), "spaced", step=15)

    print("Processing Method 6...")
    preprocess_frames(train_files, train_audio_dir, os.path.join(output_train_dir, "method_6_offset"), "continuous", n_frames=100, start=100)
    preprocess_frames(test_files, test_audio_dir, os.path.join(output_test_dir, "method_6_offset"), "continuous", n_frames=100, start=100)

    print("Preprocessing complete. All datasets saved.")
