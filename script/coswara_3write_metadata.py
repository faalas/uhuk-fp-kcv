import os
import librosa
import pandas as pd

def generate_cleaned_metadata(cleaned_folder, output_csv_path):
    data = []
    for label in ["pos", "neg"]:
        folder_path = os.path.join(cleaned_folder, label)
        if not os.path.exists(folder_path):
            print(f"Folder tidak ditemukan: {folder_path}")
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    data.append({
                        "filename": filename,
                        "duration": duration,
                        "label": label
                    })
                except Exception as e:
                    print(f"Gagal memproses {file_path}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Metadata berhasil disimpan ke {output_csv_path}")

# Script utama
if __name__ == "__main__":
    generate_cleaned_metadata(
        cleaned_folder="Dataset Coswara Segmented",
        output_csv_path="Dataset Coswara Segmented/metadata_segmented.csv"
    )
