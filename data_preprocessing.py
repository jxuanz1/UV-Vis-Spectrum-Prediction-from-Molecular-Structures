import pandas as pd

# 加载并预处理数据
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print('Data loaded successfully.')
    print('First 5 rows of data:')
    print(df.head())
    print('\nData information:')
    df.info()
    print('\nMissing values statistics:')
    print(df.isnull().sum())
    
    # 提取所有列名为数字的波长数据列
    wavelength_cols = [col for col in df.columns if col.isdigit()]
    df[wavelength_cols] = df[wavelength_cols].astype(float)
    
    # 显示光谱边界信息说明
    print('\nMeaning of lbound and ubound columns:')
    print('lbound: starting wavelength of the measured spectrum (nm)')
    print('ubound: ending wavelength of the measured spectrum (nm)')
    print('Some spectra are extrapolated, which means lbound and ubound may not span the full 200-800 nm range.')
    
    return df, wavelength_cols

if __name__ == '__main__':
    file_path = r"C:\Users\32258\Desktop\CHEMAI\UV\UVvisdata_200-800_8379.csv"
    df, wavelength_cols = load_and_preprocess_data(file_path)
    print(f'\nThe dataset contains {len(wavelength_cols)} wavelength points, ranging from {wavelength_cols[0]}nm to {wavelength_cols[-1]}nm.')
