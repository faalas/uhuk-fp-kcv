# app.py
import os
import gradio as gr
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf
from scipy.signal import lfilter
from copy import deepcopy
import speechproc  # make sure this file is available

# === Load Model ===
model = tf.keras.models.load_model('model.keras')

yamnet_model = None
class_names = None

def load_model_yamnet():
    global yamnet_model, class_names
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_names = list(pd.read_csv('yamnet_class_map.csv')['display_name'])

# === Voice Activity Detection (VAD) Function ===
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

    noise_samp, _, n_noise_samp = speechproc.snre_highenergy(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk
    )
    for j in range(n_noise_samp):
        fdata[round(noise_samp[j, 0]): round(noise_samp[j, 1]) + 1] = 0

    vad_seg = speechproc.snre_vad(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres
    )
    return vad_seg

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

def validate_cough(file_path):
    global yamnet_model, class_names
    if yamnet_model is None or class_names is None:
        load_model_yamnet()

    try:
        waveform, sr = librosa.load(file_path, sr=16000)
        waveform = waveform.astype(np.float32)

        scores, embeddings, spectrogram = yamnet_model(waveform)
        scores_np = scores.numpy()

        cough_index = class_names.index('Cough')
        cough_score = float(np.max(scores_np[:, cough_index]))
        is_cough = 1 if cough_score > 0.5 else 0

        mean_scores = np.mean(scores_np, axis=0)

        return cough_score, is_cough

    except Exception as e:
        return 0.0, 0, "Unknown"

def extract_features(segments, sr):
    X_mfcc, X_tonnetz, X_mel, X_contrast, X_chroma = [], [], [], [], []

    for seg in segments:
        if len(seg) == 0:
            continue

        seg = np.array(seg)
        seg = seg.astype(np.float32)
        mfcc = np.mean(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=40), axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(seg), sr=sr), axis=1)
        mel = np.mean(librosa.feature.melspectrogram(y=seg, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=seg, sr=sr), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=seg, sr=sr), axis=1)

        X_mfcc.append(mfcc)
        X_tonnetz.append(tonnetz)
        X_mel.append(mel)
        X_contrast.append(contrast)
        X_chroma.append(chroma)

    X = np.array([
        np.concatenate([X_mfcc[i], X_tonnetz[i], X_mel[i], X_contrast[i], X_chroma[i]])
        for i in range(len(X_mfcc))
    ])

    return X

def predict_audio(file_path):
    segments, sr = segmentation(file_path)
    results = []
    cough_scores = []
    
    # Tahap 1: Validasi batuk untuk setiap segmen
    for i, seg in enumerate(segments):
        temp_file = f"temp_seg_{i}.wav"
        sf.write(temp_file, np.array(seg, dtype='float32'), sr)
        cough_score, is_cough = validate_cough(temp_file)
        os.remove(temp_file)
        
        # Default 0, jika validasi batuk >0.5 tetap 0, else -1
        status = 0 if is_cough else -1
        cough_scores.append(status)
    
    # Jika semua segmen bukan batuk, langsung return
    if all(score == -1 for score in cough_scores):
        return []
    
    # Tahap 2: Ekstraksi fitur hanya untuk segmen batuk (status 0)
    valid_segments = [seg for seg, score in zip(segments, cough_scores) if score == 0]
    valid_indices = [i for i, score in enumerate(cough_scores) if score == 0]
    
    if not valid_segments:
        return []
    
    features = extract_features(valid_segments, sr)
    
    # Tahap 3: Prediksi untuk segmen batuk
    for i, (seg_idx, seg, feat) in enumerate(zip(valid_indices, valid_segments, features)):
        if len(feat) == 0:
            continue

        feat = np.array(feat, dtype=np.float32).reshape((1, 1, len(feat)))

        pred = model.predict(feat)[0][0]
        # Update status: 1 untuk COVID, 0 untuk NON-COVID
        cough_scores[seg_idx] = 1 if pred > 0.5 else 0
    
    return cough_scores

def predict_demo(file_path):
    if not file_path.lower().endswith('.wav'):
        return "⚠️ Mohon unggah file dengan format **.wav** saja."
    
    results = predict_audio(file_path)
    
    if not results:  # Jika hasil kosong (tidak ada segmen batuk)
        return "🚫 Tidak ada segmen batuk yang valid."
    
    # Format output
    output_lines = [f"Segment {i+1}: {'COVID' if score == 1 else 'NON-COVID' if score == 0 else 'Bukan Batuk'}" 
                   for i, score in enumerate(results)]
    
    # Hitung jumlah masing-masing kategori
    covid_count = results.count(1)
    non_covid_count = results.count(0)
    non_cough_count = results.count(-1)
    
    # Buat keputusan akhir
    final_decision = "COVID" if 1 in results else "NON-COVID" if 0 in results else "TIDAK ADA BATUK"
    
    summary = [
        "\n=== 🧾 Ringkasan ===",
        f"Jumlah Segmen COVID     : {covid_count}",
        f"Jumlah Segmen NON-COVID : {non_covid_count}",
        f"Jumlah Segmen Bukan Batuk: {non_cough_count}",
        f"\n🧠 Hasil Akhir: File ini lebih cenderung ke: **{final_decision}**"
    ]
    
    return "\n".join(output_lines + summary)

# === Gradio Interface ===
gr.Interface(
    fn=predict_demo,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs="text",
    title="COVID-19 Cough Detector",
    description="Upload or record a cough to predict whether it's COVID or not."
).launch()