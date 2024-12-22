import os
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch
from tqdm import tqdm

# Paths
train_original_dir = "data/train/original/"
augmented_dir = "data/train/augmented_signals/"

# Ensure the augmented directory exists
os.makedirs(augmented_dir, exist_ok=True)

# Data augmentation pipeline
audio_augment = Compose([
    AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.15, p=1),
    TimeStretch(min_rate=0.75, max_rate=1.25, p=1)
])

# Augment training data
print("Starting augmentation...")
for file_name in tqdm(os.listdir(train_original_dir)):
    # Load original audio
    file_path = os.path.join(train_original_dir, file_name)
    y, sr = librosa.load(file_path, sr=None)
    
    # Apply augmentation
    augmented_audio = audio_augment(samples=y, sample_rate=sr)
    
    # Generate augmented file name
    base_name, ext = os.path.splitext(file_name)
    parts = base_name.split("-")
    augmented_file_name = f"{'-'.join(parts[:-1])}_aug-{parts[-1]}{ext}"
    augmented_file_path = os.path.join(augmented_dir, augmented_file_name)
    
    # Save augmented audio
    sf.write(augmented_file_path, augmented_audio, sr)

print(f"Augmented signals saved to {augmented_dir}")
