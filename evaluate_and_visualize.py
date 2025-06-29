import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Step 1: 配置路径 ===
DATA_DIR = r"C:\Users\32258\Desktop\CHEMAI\UV222"

# === 加载数据与模型参数 ===
with open(os.path.join(DATA_DIR, "processed_data.pkl"), "rb") as f:
    df = pickle.load(f)

with open(os.path.join(DATA_DIR, "char_to_index.pkl"), "rb") as f:
    char_to_index = pickle.load(f)

with open(os.path.join(DATA_DIR, "max_smiles_length.pkl"), "rb") as f:
    max_smiles_length = pickle.load(f)

# === 加载已训练好的 .h5 模型（不编译） ===
lstm_model = tf.keras.models.load_model(os.path.join(DATA_DIR, "lstm_model.h5"), compile=False)
seq2seq_model = tf.keras.models.load_model(os.path.join(DATA_DIR, "seq2seq_model.h5"), compile=False)

# === Step 2: 准备测试数据 ===
wavelength_columns = [str(i) for i in range(200, 801)]
y = df[wavelength_columns].values.astype(np.float32)

# 压缩指纹解码与归一化
compressed_fp = df["compressed_ecfp6_fingerprints"].tolist()
X_fp = np.array(compressed_fp, dtype=np.float64)
X_fp[X_fp > 1e9] = 0.0
X_fp = np.nan_to_num(X_fp, nan=0.0, posinf=0.0, neginf=0.0)

min_vals = X_fp.min(axis=0, keepdims=True)
max_vals = X_fp.max(axis=0, keepdims=True)
X_fp_norm = (X_fp - min_vals) / (max_vals - min_vals + 1e-8)
X_fp_final = X_fp_norm.astype(np.float32).reshape(X_fp_norm.shape[0], X_fp_norm.shape[1], 1)

# SMILES 填充
X_smiles = pad_sequences(df["encoded_smiles"].tolist(), maxlen=max_smiles_length, padding="post")

# 划分测试集
X_fp_train, X_fp_test, y_train, y_test = train_test_split(X_fp_final, y, test_size=0.2, random_state=42)
X_smiles_train, X_smiles_test, _, _ = train_test_split(X_smiles, y, test_size=0.2, random_state=42)

# 构造解码器输入
decoder_input_test = np.zeros_like(y_test)
decoder_input_test[:, 1:] = y_test[:, :-1]
decoder_input_test[:, 0] = -1.0
decoder_input_test = decoder_input_test.reshape(decoder_input_test.shape[0], decoder_input_test.shape[1], 1)

# === Step 3: 模型预测 ===
y_pred_lstm = lstm_model.predict(X_fp_test)
y_pred_seq2seq = seq2seq_model.predict([X_smiles_test, decoder_input_test])

y_pred_lstm = np.nan_to_num(y_pred_lstm)
y_pred_seq2seq = np.nan_to_num(y_pred_seq2seq)

# === Step 4: 评估 MAE ===
mae_lstm = mean_absolute_error(y_test.flatten(), y_pred_lstm.flatten())
mae_seq2seq = mean_absolute_error(y_test.flatten(), y_pred_seq2seq.flatten())

print(f"LSTM MAE: {mae_lstm:.4f}")
print(f"Seq2Seq MAE: {mae_seq2seq:.4f}")

# === Step 5: 可视化前 5 个样本 ===
wavelengths = np.arange(200, 801)
fig, axs = plt.subplots(5, 2, figsize=(14, 15))

for i in range(5):
    axs[i, 0].plot(wavelengths, y_test[i], label="True")
    axs[i, 0].plot(wavelengths, y_pred_lstm[i], '--', label="LSTM Pred")
    axs[i, 0].set_title(f"LSTM Prediction - Sample {i}")
    axs[i, 0].legend()

    axs[i, 1].plot(wavelengths, y_test[i], label="True")
    axs[i, 1].plot(wavelengths, y_pred_seq2seq[i], '--', label="Seq2Seq Pred")
    axs[i, 1].set_title(f"Seq2Seq Prediction - Sample {i}")
    axs[i, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "prediction_comparison.png"))
plt.show()
