import numpy as np
import matplotlib.pyplot as plt

# Load data dan label
X = np.load('features/X_mfcc.npy')     # shape: (n_samples, 40)
y = np.load('features/y.npy')          # shape: (n_samples,)

print(X)

# Pisahkan data berdasarkan label
X_pos = X[y == 1]
X_neg = X[y == 0]

print(f"Data Shape: {X.shape}")
print(f"Data Type: {X.dtype}")

print(f"Pos Shape: {X_pos.shape}")
print(f"Pos Type: {X_pos.dtype}")

print(f"Neg Shape: {X_neg.shape}")
print(f"Neg Type: {X_neg.dtype}")

# --- Histogram per label ---
plt.figure(figsize=(10, 5))
plt.hist(X_neg.flatten(), bins=50, alpha=0.5, label='Negatif', color='red')
plt.hist(X_pos.flatten(), bins=50, alpha=0.5, label='Positif', color='green')
plt.title('Distribusi Fitur MFCC (Pos vs Neg)')
plt.xlabel('Nilai Fitur')
plt.ylabel('Frekuensi')
plt.legend()
plt.show()

# --- Heatmap: rata-rata per label ---
plt.figure(figsize=(12, 5))

# Hitung rata-rata MFCC
mean_neg = np.mean(X_neg, axis=0, keepdims=True)
mean_pos = np.mean(X_pos, axis=0, keepdims=True)

# Cari skala warna bersama
vmin = min(mean_neg.min(), mean_pos.min())
vmax = max(mean_neg.max(), mean_pos.max())

plt.subplot(1, 2, 1)
plt.imshow(mean_neg, aspect='auto', origin='lower', cmap='seismic', vmin=vmin, vmax=vmax)
plt.title('Rata-rata MFCC - Negatif')
plt.xlabel('Koefisien MFCC')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(mean_pos, aspect='auto', origin='lower', cmap='seismic', vmin=vmin, vmax=vmax)
plt.title('Rata-rata MFCC - Positif')
plt.xlabel('Koefisien MFCC')
plt.colorbar()

plt.tight_layout()
plt.show()