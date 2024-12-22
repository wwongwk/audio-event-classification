import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Define paths
metadata_path = "data/raw/meta/esc50.csv"
audio_dir = "data/raw/audio/"
train_dir = "data/train/original/"
test_dir = "data/test/original/"

# Create train/test directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Clear existing files in train/test directories
for folder in [train_dir, test_dir]:
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        os.remove(file_path)

# Load metadata
metadata = pd.read_csv(metadata_path)

# Check for missing files in audio_dir
missing_files = [f for f in metadata['filename'] if not os.path.exists(os.path.join(audio_dir, f))]
if missing_files:
    print(f"Warning: {len(missing_files)} files are missing in {audio_dir}.")
    metadata = metadata[~metadata['filename'].isin(missing_files)]

# Stratified split
train_df, test_df = train_test_split(metadata, test_size=0.2, stratify=metadata['target'], random_state=42)

# Copy files to train/test directories
print("Copying training files...")
for _, row in train_df.iterrows():
    src = os.path.join(audio_dir, row['filename'])
    dest = os.path.join(train_dir, row['filename'])
    shutil.copy(src, dest)

print("Copying testing files...")
for _, row in test_df.iterrows():
    src = os.path.join(audio_dir, row['filename'])
    dest = os.path.join(test_dir, row['filename'])
    shutil.copy(src, dest)

print(f"Dataset split completed:\n- Training files: {len(train_df)}\n- Testing files: {len(test_df)}")
