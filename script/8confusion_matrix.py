import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score,
    precision_recall_curve
)

from tensorflow.keras.models import load_model

# Load model
model = load_model("model2_10.keras")

# Evaluasi akurasi
scores = model.evaluate(val_x, val_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# --- Prediksi pada validation set ---
y_pred_probs = model.predict(val_x)
y_pred_labels = (y_pred_probs > 0.5).astype(int)

# --- Confusion Matrix ---
cm = confusion_matrix(val_y, y_pred_labels)
tn, fp, fn, tp = cm.ravel()

# --- Metrics ---
accuracy     = accuracy_score(val_y, y_pred_labels)
recall       = recall_score(val_y, y_pred_labels)   # Sensitivity
precision    = precision_score(val_y, y_pred_labels)
f1           = f1_score(val_y, y_pred_labels)
specificity  = tn / (tn + fp)
uar          = (recall + specificity) / 2           # Unweighted Average Recall
bias         = abs(sum(y_pred_labels.flatten()) / len(y_pred_labels) - sum(val_y) / len(val_y))  # Prediksi vs label distribusi
inference_time = end_time - start_time              # Waktu total training

# --- Tampilkan semua metrik ---
print(f"Akurasi       : {accuracy:.4f}")
print(f"Recall        : {recall:.4f} (Sensitivitas)")
print(f"Precision     : {precision:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"Specificity   : {specificity:.4f}")
print(f"UAR           : {uar:.4f}")
print(f"Bias Testing  : {bias:.4f}")
print(f"Waktu Komputasi Training: {inference_time:.2f} detik")

auc = roc_auc_score(val_y, y_pred_labels)
print(f"AUC: {auc:.2f}")

# --- Visualisasi Confusion Matrix ---
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

# For Precision Improvement:
precision, recall, thresholds = precision_recall_curve(val_y, y_pred_labels)
optimal_idx = np.argmax(precision * recall)  # F1-score maximization
optimal_threshold = thresholds[optimal_idx]