import os
import librosa
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# Folder sumber data dan label
coswara = ["Dataset Coswara Cleaned"]
coughvid = ["Dataset Cough Covid Cleaned"]
root_dirs = coswara + coughvid  # Menggabungkan kedua folder sumber

labels_map = {'pos': 1, 'neg': 0}

def extract_feature(X, sample_rate):
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate), axis=1)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate), axis=1)
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate, n_fft=275), axis=1)
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate, n_fft=275), axis=1)
    flatness = np.mean(librosa.feature.spectral_flatness(y=X, n_fft=275), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=X, sr=sample_rate, n_fft=275), axis=1)

    return chroma, mel, contrast, tonnetz, mfcc, centroid, bandwidth, flatness, rolloff

def process_file(args):
    file_path, label, root_dir = args
    try:
        X, sr = librosa.load(file_path, sr=None)
        features = extract_feature(X, sr)
        if features is None:
            return None
        chroma, mel, contrast, tonnetz, mfcc, centroid, bandwidth, flatness, rolloff = features
        features_dict = {
            'mfcc': mfcc,
            'chroma': chroma,
            'mel': mel,
            'contrast': contrast,
            'tonnetz': tonnetz,
            'centroid': centroid,
            'bandwidth': bandwidth,
            'flatness': flatness,
            'rolloff': rolloff,
            'label': labels_map[label],
            'source': root_dir
        }
        return features_dict
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def collect_files():
    """Kumpulkan semua file .wav beserta label dan folder sumbernya"""
    all_files = []
    for root_dir in root_dirs:
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    if file_name.endswith(".wav"):
                        file_path = os.path.join(label_dir, file_name)
                        all_files.append((file_path, label, root_dir))
    return all_files

def main():
    all_files = collect_files()

    results = []
    with Pool(processes=15) as pool:
        for res in tqdm(pool.imap_unordered(process_file, all_files), total=len(all_files), desc="Extracting features", unit="file"):
            if res is not None:
                results.append(res)

    # Ekstrak fitur menjadi array
    feature_keys = ['mfcc', 'chroma', 'mel', 'contrast', 'tonnetz', 'centroid', 'bandwidth', 'flatness', 'rolloff']
    feature_arrays = {key: [r[key] for r in results] for key in feature_keys}
    y_all = [r['label'] for r in results]
    sources = [r['source'] for r in results]

    # Simpan hasil
    os.makedirs("features", exist_ok=True)
    for key in feature_keys:
        np.save(f"features/X_{key}.npy", np.array(feature_arrays[key]))
    np.save("features/y.npy", np.array(y_all))
    np.save("features/source.npy", np.array(sources))

if __name__ == "__main__":
    main()