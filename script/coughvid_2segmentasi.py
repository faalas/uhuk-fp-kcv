import os
import math
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from copy import deepcopy
from scipy.signal import lfilter
import multiprocessing as mp

import speechproc

# GLOBAL metadata yang akan dibaca di tiap worker
df_meta = None

# Fungsi inisialisasi worker (memuat metadata sekali saja per proses)
def init_worker(metadata_path):
    global df_meta
    df_meta = pd.read_csv(metadata_path)

# Fungsi untuk mendeteksi Voice Activity (VAD)
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

# Fungsi utama segmentasi
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

    np_list_X = np.array(list_X, dtype=object)
    return np_list_X, sample_rate

# Fungsi pemrosesan paralel per file
def process_and_save_segment(args):
    filename, src_folder, dst_folder, status_col, checkpoint_path = args
    try:
        name = os.path.splitext(filename)[0]
        row = df_meta[df_meta.iloc[:, 0].str.contains(name)]
        if row.empty:
            return

        label = row.iloc[0, status_col]
        kategori = "pos" if str(label).strip().lower() == "covid-19" else "neg"

        file_path = os.path.join(src_folder, filename)
        segments, sr = segmentation(file_path)

        for idx, segment in enumerate(segments):
            outname = f"{name}_seg{idx}.wav"
            outpath = os.path.join(dst_folder, kategori, outname)
            sf.write(outpath, np.asarray(segment, dtype=np.float32), sr)

        # Simpan checkpoint setelah sukses
        with open(checkpoint_path, "a") as f:
            f.write(f"{filename}\n")

    except Exception as e:
        print(f"Gagal proses {filename}: {e}")

# Fungsi paralel utama
def parallel_save_segments_to_folder(
    src_folder,
    dst_folder,
    metadata_path,
    status_col=10,
    num_workers=8,
    checkpoint_path="coughvid_checkpoint.txt"
):
    os.makedirs(os.path.join(dst_folder, "pos"), exist_ok=True)
    os.makedirs(os.path.join(dst_folder, "neg"), exist_ok=True)

    # Load file checkpoint
    processed_files = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            processed_files = set(line.strip() for line in f)

    # Siapkan file untuk diproses
    wav_files = [f for f in os.listdir(src_folder) if f.endswith(".wav") and f not in processed_files]
    args = [(f, src_folder, dst_folder, status_col, checkpoint_path) for f in wav_files]

    if not wav_files:
        print("Semua file sudah diproses. Tidak ada yang perlu dilakukan.")
        return

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=init_worker, initargs=(metadata_path,)) as pool:
        list(tqdm(pool.imap_unordered(process_and_save_segment, args), total=len(wav_files), desc="Segmenting & saving (parallel)"))

# Main script
if __name__ == "__main__":
    parallel_save_segments_to_folder(
        src_folder="Dataset Cough Covid WAV",
        dst_folder="Dataset Cough Covid Segmented",
        metadata_path="Dataset Cough Covid/metadata_compiled.csv",
        status_col=10,
        num_workers=16,
        checkpoint_path="coughvid_checkpoint.txt"
    )