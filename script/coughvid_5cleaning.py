# coughvid_cleaning.py
import os
import shutil
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Queue, current_process

segmented_base = 'Dataset Cough Covid Segmented'
cleaned_base = 'Dataset Cough Covid Cleaned'
metadata_path = os.path.join(segmented_base, 'metadata_segmented.csv')

df = pd.read_csv(metadata_path)
df_valid = df[df['is_cough'] == 1].copy()
df_valid.reset_index(drop=True, inplace=True)

for label in ['pos', 'neg']:
    os.makedirs(os.path.join(cleaned_base, label), exist_ok=True)

def worker(copy_list, queue):
    for _, row in copy_list.iterrows():
        label = row['label']
        filename = row['filename']
        src_path = os.path.join(segmented_base, label, filename)
        dst_path = os.path.join(cleaned_base, label, filename)
        try:
            shutil.copy2(src_path, dst_path)
            queue.put(1)
        except Exception as e:
            print(f"[{current_process().name}] Gagal menyalin {filename}: {e}")
            queue.put(1)

def split_dataframe(df, n):
    return [df.iloc[i::n] for i in range(n)]

def run_multiprocessing_copy(df_valid, num_workers=10):
    queue = Queue()
    processes = []
    df_splits = split_dataframe(df_valid, num_workers)

    for i in range(num_workers):
        p = Process(target=worker, args=(df_splits[i], queue), name=f"Worker-{i+1}")
        p.start()
        processes.append(p)

    with tqdm(total=len(df_valid), desc="Menyalin audio valid") as pbar:
        copied = 0
        while copied < len(df_valid):
            queue.get()
            copied += 1
            pbar.update(1)

    for p in processes:
        p.join()

if __name__ == '__main__':
    run_multiprocessing_copy(df_valid, num_workers=10)
    metadata_cleaned_path = os.path.join(cleaned_base, 'metadata_cleaned.csv')
    df_valid.to_csv(metadata_cleaned_path, index=False)
    print(f"✅ Metadata cleaned disimpan ke: {metadata_cleaned_path}")