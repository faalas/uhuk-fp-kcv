# coswara_script_extract.py
# Download aja smw dlu yg di github
# Bikin file py baruuu trs run
# tar.gz -> wav
# Ekstrak semua .tar.gz.* dari Dataset Coswara ke Dataset Coswara Extracted

import os
import glob
import tarfile

'''
Script ini akan mengekstrak file .tar.gz.* dari folder "Dataset Coswara"
dan menyimpan hasilnya ke "Dataset Coswara Extracted", dipisah berdasarkan tanggal.
'''

# Path ke Dataset Coswara (input)
coswara_data_dir = os.path.join(os.path.dirname(__file__), 'Dataset Coswara')
# Path ke folder hasil ekstraksi
extracted_data_dir = os.path.join(os.path.dirname(__file__), 'Dataset Coswara Extracted')

# Validasi folder sumber
if not os.path.exists(coswara_data_dir):
    raise FileNotFoundError("Folder 'Dataset Coswara' tidak ditemukan!")

# Buat folder tujuan kalau belum ada
if not os.path.exists(extracted_data_dir):
    os.makedirs(extracted_data_dir)

# Ambil nama folder tanggal (202*) dari sumber dan hasil
dirs_all = set(map(os.path.basename, glob.glob(f'{coswara_data_dir}/202*')))
dirs_extracted = set(map(os.path.basename, glob.glob(f'{extracted_data_dir}/202*')))

# Ambil yang belum diproses
dirs_to_extract = list(dirs_all - dirs_extracted)

for d in dirs_to_extract:
    folder_path = os.path.join(coswara_data_dir, d)
    parts = sorted(glob.glob(os.path.join(folder_path, '*.tar.gz.*')))

    if not parts:
        print(f"[!] Tidak ditemukan file tar.gz di {d}, skip.")
        continue

    print(f"[+] Memproses {d} ...")

    # Gabungkan bagian-bagian .tar.gz
    combined_path = os.path.join(extracted_data_dir, f'{d}.tar.gz')
    with open(combined_path, 'wb') as f_out:
        for part in parts:
            with open(part, 'rb') as f_in:
                f_out.write(f_in.read())

    # Ekstrak hasilnya
    with tarfile.open(combined_path, 'r:gz') as tar:
        tar.extractall(path=os.path.join(extracted_data_dir, d))

    # Hapus file gabungan
    os.remove(combined_path)

print("✅ Proses ekstraksi selesai!")