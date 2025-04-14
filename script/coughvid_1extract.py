import os
import subprocess
from tqdm import tqdm

# Konversi .webm ke .wav
def convert_webm_to_wav(input_path, output_path, target_sr=22050):
    ffmpeg_path = os.path.join("ffmpeg", "bin", "ffmpeg.exe")
    command = [
        ffmpeg_path,
        "-i", input_path,
        "-ar", str(target_sr),
        "-ac", "1",
        output_path,
        "-y"  # overwrite jika sudah ada
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def convert_all_webm_to_wav(src_folder, dst_folder, target_sr=22050):
    os.makedirs(dst_folder, exist_ok=True)
    webm_files = [f for f in os.listdir(src_folder) if f.endswith(".webm")]

    for filename in tqdm(webm_files, desc="Converting webm to wav"):
        input_path = os.path.join(src_folder, filename)
        output_path = os.path.join(dst_folder, filename.replace(".webm", ".wav"))
        convert_webm_to_wav(input_path, output_path, target_sr)

def convert_all_webm_to_wav(src_folder, dst_folder, target_sr=22050):
    os.makedirs(dst_folder, exist_ok=True)
    webm_files = [f for f in os.listdir(src_folder) if f.endswith(".webm")]

    for filename in tqdm(webm_files, desc="Converting webm to wav"):
        input_path = os.path.join(src_folder, filename)
        output_path = os.path.join(dst_folder, filename.replace(".webm", ".wav"))
        convert_webm_to_wav(input_path, output_path, target_sr)

# Script utama
if __name__ == "__main__":
    convert_all_webm_to_wav("Dataset Cough Covid", "Dataset Cough Covid WAV")