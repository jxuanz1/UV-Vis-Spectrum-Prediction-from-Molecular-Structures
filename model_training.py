import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load preprocessed data
with open(r"C:\Users\32258\Desktop\CHEMAI\UV222\processed_data.pkl", "rb") as f:
    df = pickle.load(f)

# 准备输入特征和目标光谱
wavelength_columns = [str(i) for i in range(200, 801)]
X_fingerprint = np.array(df["compressed_ecfp6_fingerprints"].tolist(), dtype=np.float32)
y_spectrum = df[wavelength_columns].values.astype(np.float32)

# 重塑指纹为 LSTM 输入格式：(samples, timesteps, features)
X_fingerprint = X_fingerprint.reshape(X_fingerprint.shape[0], X_fingerprint.shape[1], 1)

# 获取最大 SMILES 长度和词汇表
encoded_smiles_list = df["encoded_smiles"].tolist()
max_smiles_length = max(len(seq) for seq in encoded_smiles_list)

# 从 feature_engineering 中导入 SMILES 编码
from feature_engineering import smiles_to_int_encoding
_, char_to_index = smiles_to_int_encoding(df["SMILES"].tolist())

# 设置词汇表大小（防止索引越界）
vocab_size = max(char_to_index.values()) + 1

# 验证是否存在非法索引
max_smiles_index = max([max(seq) for seq in encoded_smiles_list if len(seq) > 0])
print("Max index in encoded SMILES:", max_smiles_index)
print("Vocabulary size:", vocab_size)
assert max_smiles_index < vocab_size, "Index out of vocabulary range detected in encoded SMILES!"

# 对 SMILES 进行填充
X_smiles_padded = pad_sequences(encoded_smiles_list, maxlen=max_smiles_length, padding='post')

# 划分训练测试集
X_fp_train, X_fp_test, y_train, y_test = train_test_split(X_fingerprint, y_spectrum, test_size=0.2, random_state=42)
X_smiles_train, X_smiles_test, _, _ = train_test_split(X_smiles_padded, y_spectrum, test_size=0.2, random_state=42)

# 构建 LSTM 模型（基于 ECFP6 指纹）
def build_lstm_model(input_shape, output_dim, dropout_rate=0.3):
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        LSTM(2048, activation='relu', return_sequences=True),
        Dropout(dropout_rate),
        LSTM(1024, activation='relu', return_sequences=True),
        Dropout(dropout_rate),
        LSTM(512, activation='relu', return_sequences=True),
        Dropout(dropout_rate),
        LSTM(156, activation='relu'),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dense(output_dim)
    ])
    return model

# 构建 Seq2Seq 模型（基于 SMILES + 谱图）
def build_seq2seq_model(vocab_size, max_smiles_length, output_dim, embedding_dim=1024, latent_dim=512):
    # 编码器部分
    encoder_inputs = Input(shape=(max_smiles_length,))
    encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
    enc_out1, h1, c1 = encoder_lstm1(encoder_embedding)
    encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
    enc_out2, h2, c2 = encoder_lstm2(enc_out1)
    encoder_lstm3 = LSTM(latent_dim, return_state=True)
    _, final_h, final_c = encoder_lstm3(enc_out2)
    encoder_states = [final_h, final_c]

    # 解码器部分
    decoder_inputs = Input(shape=(output_dim, 1))
    decoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
    dec_out1, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
    dec_out2, _, _ = decoder_lstm2(dec_out1)
    decoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True)
    dec_out3, _, _ = decoder_lstm3(dec_out2)
    decoder_lstm4 = LSTM(latent_dim, return_sequences=True, return_state=True)
    dec_out4, _, _ = decoder_lstm4(dec_out3)
    decoder_lstm5 = LSTM(latent_dim, return_sequences=True, return_state=True)
    dec_out5, _, _ = decoder_lstm5(dec_out4)
    decoder_lstm6 = LSTM(latent_dim, return_sequences=True, return_state=True)
    dec_out6, _, _ = decoder_lstm6(dec_out5)

    # 输出层：每个时间步输出一个吸光度值
    decoder_dense = Dense(1, activation='linear')
    decoder_outputs = decoder_dense(dec_out6)
    decoder_outputs = tf.keras.layers.Reshape((output_dim,))(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

output_dim = len(wavelength_columns)

# 训练 LSTM 模型
print("Training LSTM model...")
lstm_model = build_lstm_model(X_fp_train.shape[1:], output_dim)
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
early_stop_lstm = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit(X_fp_train, y_train, epochs=300, batch_size=10, validation_split=0.2, callbacks=[early_stop_lstm])
lstm_model.save('lstm_model.keras')
print("LSTM model training completed and saved.")

# 构造解码器输入（teacher forcing）
print("Training Seq2Seq model...")
decoder_input = np.zeros_like(y_train)
decoder_input[:, 1:] = y_train[:, :-1]
decoder_input[:, 0] = -1.0  # 起始标志
decoder_input = decoder_input.reshape(decoder_input.shape[0], decoder_input.shape[1], 1)

# 构建并训练 Seq2Seq 模型
seq2seq_model = build_seq2seq_model(vocab_size, max_smiles_length, output_dim)
seq2seq_model.compile(optimizer=Adam(learning_rate=1e-4), loss='mae')
early_stop_seq2seq = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

seq2seq_model.fit(
    [X_smiles_train, decoder_input],
    y_train,
    epochs=115,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop_seq2seq]
)
seq2seq_model.save('seq2seq_model.keras')
print("Seq2Seq model training completed and saved.")

# 保存 SMILES 映射表和最大长度
with open('char_to_index.pkl', 'wb') as f:
    pickle.dump(char_to_index, f)
with open('max_smiles_length.pkl', 'wb') as f:
    pickle.dump(max_smiles_length, f)

print("Vocabulary and max SMILES length saved.")
