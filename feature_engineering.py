import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # 禁用 RDKit 的警告信息

# 生成 ECFP6 指纹（2048位）
def generate_ecfp6_fingerprints(smiles_list):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
            fingerprints.append([int(bit) for bit in fp.ToBitString()])  # 将 '0'/'1' 转换为 0/1 整数
        else:
            fingerprints.append([0] * 2048)  # 处理无效的 SMILES
    return pd.DataFrame(fingerprints, dtype=int)

# 将 ECFP6 指纹压缩为每组256位转换为一个十进制整数
def compress_ecfp6_fingerprints(fingerprints_df):
    compressed_fps = []
    for _, row in fingerprints_df.iterrows():
        compressed_row = []
        for i in range(0, 2048, 256):
            group = "".join(map(str, row[i:i+256].tolist()))
            compressed_row.append(int(group, 2))  # 将二进制转换为十进制整数
        compressed_fps.append(compressed_row)
    return pd.DataFrame(compressed_fps)

# 将 SMILES 字符串按字符编码为整数序列
def smiles_to_int_encoding(smiles_list):
    # 收集所有唯一字符
    all_chars = set()
    for smiles in smiles_list:
        for char in smiles:
            all_chars.add(char)
    
    # 添加特殊标记符号
    all_chars.add("<B>")     # 序列开始
    all_chars.add("<EOS>")   # 序列结束
    
    # 构建字符到整数的映射字典
    char_to_int = {char: i + 1 for i, char in enumerate(sorted(list(all_chars)))}
    char_to_int["<B>"] = 0  # 序列开始符号设为0
    
    # 编码每个SMILES字符串
    encoded_smiles = []
    for smiles in smiles_list:
        encoded_seq = [char_to_int["<B>"]] + [char_to_int[char] for char in smiles] + [char_to_int["<EOS>"]]
        encoded_smiles.append(encoded_seq)
    
    return encoded_smiles, char_to_int

if __name__ == '__main__':
    file_path = r"C:\Users\32258\Desktop\CHEMAI\UV\UVvisdata_200-800_8379.csv"
    df = pd.read_csv(file_path)
    
    smiles_list = df['SMILES'].tolist()
    
    print('Generating ECFP6 fingerprints...')
    ecfp6_fps_df = generate_ecfp6_fingerprints(smiles_list)
    print('First 5 rows of ECFP6 fingerprints:')
    print(ecfp6_fps_df.head())
    print(f'Shape of ECFP6 fingerprints: {ecfp6_fps_df.shape}')
    
    print('\nCompressing ECFP6 fingerprints...')
    compressed_ecfp6_fps_df = compress_ecfp6_fingerprints(ecfp6_fps_df)
    print('First 5 rows of compressed ECFP6 fingerprints:')
    print(compressed_ecfp6_fps_df.head())
    print(f'Shape of compressed ECFP6 fingerprints: {compressed_ecfp6_fps_df.shape}')
    
    print('\nEncoding SMILES to integer sequences...')
    encoded_smiles, char_to_int = smiles_to_int_encoding(smiles_list)
    print('Example encoded SMILES:')
    print(encoded_smiles[0])
    print('Size of character-to-integer mapping:', len(char_to_int))

    # 保存处理后的数据
    df['ecfp6_fingerprints'] = list(ecfp6_fps_df.values)
    df['compressed_ecfp6_fingerprints'] = list(compressed_ecfp6_fps_df.values)
    df['encoded_smiles'] = encoded_smiles
    df.to_pickle('processed_data.pkl')
    print('\nProcessed data saved to processed_data.pkl')
