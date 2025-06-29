import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

# ==== 启用交互式图像显示 ====
plt.ion()

# ==== 加载模型和预处理参数 ====
lstm_model = load_model("lstm_model.h5", compile=False)
seq2seq_model = load_model("seq2seq_model.h5", compile=False)

with open('char_to_index.pkl', 'rb') as f:
    char_to_index = pickle.load(f)

with open('max_smiles_length.pkl', 'rb') as f:
    max_smiles_length = pickle.load(f)

# ==== 功能函数 ====

def compress_ecfp6(smiles, n_bits=2048, group_size=256):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits)
    bit_string = fp.ToBitString()
    compressed = [int(bit_string[i:i+group_size], 2) for i in range(0, n_bits, group_size)]
    return np.array(compressed, dtype=np.float32).reshape(1, -1, 1)

def encode_smiles(smiles):
    encoded = []
    i = 0
    while i < len(smiles):
        if i + 1 < len(smiles) and smiles[i:i+2] in char_to_index:
            encoded.append(char_to_index[smiles[i:i+2]])
            i += 2
        elif smiles[i] in char_to_index:
            encoded.append(char_to_index[smiles[i]])
            i += 1
        else:
            return None
    return pad_sequences([encoded], maxlen=max_smiles_length, padding='post')

def predict_uv(smiles):
    print(f"\n【输入 SMILES】：{smiles}")

    ecfp6 = compress_ecfp6(smiles)
    if ecfp6 is None:
        raise ValueError("无效 SMILES：ECFP6 生成失败")
    print("ECFP6 shape:", ecfp6.shape)

    lstm_pred = lstm_model.predict(ecfp6, verbose=0)[0]
    print("LSTM 前5个值:", lstm_pred[:5])

    smiles_encoded = encode_smiles(smiles)
    if smiles_encoded is None:
        raise ValueError("SMILES 编码失败，可能包含非法字符")
    print("SMILES 编码前5个值:", smiles_encoded[0][:5])

    decoder_input = np.zeros((1, 601, 1), dtype=np.float32)
    seq2seq_pred = seq2seq_model.predict([smiles_encoded, decoder_input], verbose=0)[0]
    print("Seq2Seq 前5个值:", seq2seq_pred[:5])

    return lstm_pred, seq2seq_pred

# ==== 图像窗口初始化 ====
fig, ax = plt.subplots(figsize=(10, 5))

def show_prediction():
    smiles = entry.get().strip()
    if not smiles:
        messagebox.showwarning("输入错误", "请输入 SMILES 结构式")
        return
    try:
        lstm_pred, seq2seq_pred = predict_uv(smiles)
        wavelengths = list(range(200, 801))

        ax.clear()
        ax.plot(wavelengths, lstm_pred, label='LSTM (ECFP6)', linewidth=2)
        ax.plot(wavelengths, seq2seq_pred, label='Seq2Seq (SMILES)', linestyle='--', linewidth=2)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.set_title(f"Predicted UV-Vis Spectrum for: {smiles}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.canvas.draw()

    except Exception as e:
        print("预测错误:", str(e))
        messagebox.showerror("预测失败", f"错误原因：{str(e)}")

# ==== GUI 构建 ====
root = tk.Tk()
root.title("UV-Vis 光谱预测器")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

label = tk.Label(frame, text="输入 SMILES:")
label.pack()

entry = tk.Entry(frame, width=50)
entry.pack()

predict_button = tk.Button(frame, text="预测", command=show_prediction)
predict_button.pack(pady=10)

root.mainloop()
