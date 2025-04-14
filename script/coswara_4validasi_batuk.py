# coswara_validasi_batuk.py v2 (sr = 16000) ---> USE
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

# Supress warnings from librosa/tensorflow
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load YAMNet model and class names globally (once per process)
yamnet_model = None
class_names = None

def load_model():
    global yamnet_model, class_names
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_names = list(pd.read_csv('yamnet_class_map.csv')['display_name'])

def process_file(row):
    global yamnet_model, class_names
    if yamnet_model is None or class_names is None:
        load_model()

    label = row['label']
    filename = row['filename']
    file_path = os.path.join("Dataset Coswara Segmented", label, filename)

    try:
        waveform, sr = librosa.load(file_path, sr=16000)
        waveform = waveform.astype(np.float32)

        scores, embeddings, spectrogram = yamnet_model(waveform)
        scores_np = scores.numpy()

        # Skor validasi batuk
        cough_index = class_names.index('Cough')
        cough_score = float(np.max(scores_np[:, cough_index]))
        is_cough = 1 if cough_score > 0.5 else 0

        # Jenis suara dominan
        mean_scores = np.mean(scores_np, axis=0)
        dominant_class = class_names[int(np.argmax(mean_scores))]

        return cough_score, is_cough, dominant_class

    except Exception as e:
        print(f"Gagal memproses {file_path}: {e}")
        return 0.0, 0, "Unknown"

if __name__ == "__main__":
    # Baca metadata
    metadata_path = "Dataset Coswara Segmented/metadata_segmented.csv"
    df = pd.read_csv(metadata_path)

    print(f"🔍 Memproses {len(df)} file audio dengan multiprocessing...")

    # Multiprocessing pool
    with Pool(processes=10, initializer=load_model) as pool:
        results = list(tqdm(pool.imap(process_file, [row for _, row in df.iterrows()]), total=len(df)))

    # Pisahkan hasil
    df["validasi_batuk"] = [r[0] for r in results]
    df["is_cough"] = [r[1] for r in results]
    df["jenis_suara"] = [r[2] for r in results]

    # Simpan metadata
    df.to_csv(metadata_path, index=False)
    print("✅ Metadata berhasil diperbarui dengan kolom validasi_batuk, is_cough, dan jenis_suara.")