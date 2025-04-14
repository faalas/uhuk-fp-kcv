import os
import numpy as np
import pandas as pd
import math
import json
from multiprocessing import Pool
import multiprocessing as mp
import librosa
import soundfile as sf
from scipy.signal import lfilter
from copy import deepcopy
from tqdm import tqdm

import speechproc  # modul eksternal

# === Path konfigurasi ===
ROOT_PATH = "Dataset Coswara Extracted"
OUT_FOLDER = "Dataset Coswara Segmented"
CHECKPOINT_FILE = "coswara_checkpoint.txt"
ERROR_LOG_FILE = "coswara_error_log.txt"

def log_error(message):
    with open(ERROR_LOG_FILE, "a") as f:
        f.write(message + "\n")

os.makedirs(os.path.join(OUT_FOLDER, "pos"), exist_ok=True)
os.makedirs(os.path.join(OUT_FOLDER, "neg"), exist_ok=True)

# Fungsi untuk deteksi Voice Activity (VAD)
def getVad(finwav):
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres = 0.5
    vadThres = 0.4
    opts = 1

    fs, data = speechproc.speech_wave(finwav)

    ft, flen, fsh10, nfr10 = speechproc.sflux(data, fs, winlen, ovrlen, nftt)

    pv01 = np.zeros(nfr10)
    pv01[np.less_equal(ft, ftThres)] = 1

    pitch = deepcopy(ft)
    pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)

    ENERGYFLOOR = np.exp(-50)
    b = np.array([0.9770, -0.9770])
    a = np.array([1.0000, -0.9540])
    fdata = lfilter(b, a, data, axis=0)

    noise_samp, noise_seg, n_noise_samp = speechproc.snre_highenergy(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk
    )
    for j in range(n_noise_samp):
        fdata[round(noise_samp[j, 0]): round(noise_samp[j, 1]) + 1] = 0

    vad_seg = speechproc.snre_vad(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres
    )

    return vad_seg

# Segmentasi audio menjadi segmen suara (batuk)
def segmentation(file_name):
    X, sample_rate = librosa.load(file_name, sr=22050)
    fvad = getVad(file_name)

    list_X = []
    X_baru = []

    for i in range(1, len(fvad)):
        if fvad[i - 1] == 1:
            len_sector = math.floor(len(X) / len(fvad))
            start = (i - 1) * len_sector
            for j in range(start, start + len_sector):
                if j < len(X):
                    X_baru.append(X[j])
        if fvad[i - 1] == 1 and fvad[i] == 0:
            list_X.append(X_baru)
            X_baru = []

    return np.array(list_X, dtype=object), sample_rate

# Mapping status ke label pos/neg
def map_covid_status(status):
    status = status.strip().lower()
    if status in ["positive_mild", "positive_moderate", "positive_asymp"]:
        return "pos"
    elif status in ["healthy", "no_resp_illness_exposed", "resp_illness_not_identified", "recovered_full"]:
        return "neg"
    return None

# Ambil label dari metadata.json
def load_label_from_metadata(subdir):
    metadata_path = os.path.join(subdir, "metadata.json")
    if not os.path.exists(metadata_path):
        log_error(f"📄 metadata.json tidak ditemukan di {subdir}")
        return None

    try:
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        status = meta.get("covid_status", "")
        return map_covid_status(status)
    except Exception as e:
        log_error(f"❌ Gagal membaca metadata.json di {subdir}: {e}")
        return None

# Proses setiap folder user
def process_folder(subdir):
    user_id = os.path.basename(subdir)
    print(f"🔍 Memproses {user_id}...")
    label = load_label_from_metadata(subdir)

    if label is None:
        log_error(f"❓ Label tidak ditemukan untuk ID: {user_id}")
        return []

    hasil_meta = []
    for cough_type in ["cough-heavy.wav", "cough-shallow.wav"]:
        wav_path = os.path.join(subdir, cough_type)
        if not os.path.exists(wav_path):
            log_error(f"🚫 File tidak ditemukan: {wav_path}")
            continue

        try:
            segments, sr = segmentation(wav_path)
            for i, seg in enumerate(segments):
                seg = np.array(seg)
                if seg.size == 0 or seg.ndim == 0:
                    log_error(f"⚠️  Segmen kosong pada {wav_path} segmen ke-{i}")
                    continue
                if seg.dtype == object:
                    seg = seg.astype(np.float32)

                fname = f"{user_id}_{cough_type.replace('.wav','')}_seg{i}.wav"
                fpath = os.path.join(OUT_FOLDER, label, fname)
                sf.write(fpath, seg, sr)

                hasil_meta.append({
                    "filename": fname,
                    "duration": librosa.get_duration(y=seg, sr=sr),
                    "label": label
                })
        except Exception as e:
            log_error(f"❌ Gagal proses {wav_path}: {e}")

    with open(CHECKPOINT_FILE, "a") as fcp:
        fcp.write(subdir + "\n")

    return hasil_meta

# Main loop
def main():
    print("🚀 Memulai segmentasi dengan multiprocessing...")
    all_subdirs = []
    for subdir, dirs, files in os.walk(ROOT_PATH):
        if any(f in files for f in ["cough-heavy.wav", "cough-shallow.wav"]):
            all_subdirs.append(subdir)

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            processed = set([line.strip() for line in f])
    else:
        processed = set()

    to_process = [subdir for subdir in all_subdirs if subdir not in processed]
    print(f"📁 Total data: {len(all_subdirs)} | Belum diproses: {len(to_process)}")

    metadata_all = []
    with Pool(16) as p:
        for result in tqdm(p.imap_unordered(process_folder, to_process), total=len(to_process)):
            metadata_all.extend(result)

    if metadata_all:
        df = pd.DataFrame(metadata_all)
        df.to_csv(os.path.join(OUT_FOLDER, "metadata_segmented.csv"), index=False)
        print("✅ Segmentasi selesai! Metadata disimpan.")
    else:
        print("⚠️  Tidak ada file baru yang diproses.")

if __name__ == "__main__":
    main()