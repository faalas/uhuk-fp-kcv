import os
import librosa
import pandas as pd  

# Buat metadata hasil segmentasi
def generate_cleaned_metadata(cleaned_folder, output_csv_path):
    data = []
    for label in ["pos", "neg"]:
        folder_path = os.path.join(cleaned_folder, label)
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                data.append({
                    "filename": filename,
                    "duration": duration,
                    "label": label
                })

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)

# Script utama
if __name__ == "__main__":
    generate_cleaned_metadata(
        cleaned_folder="Dataset Cough Covid Segmented",
        output_csv_path="Dataset Cough Covid Segmented/metadata_segmented.csv"
    )