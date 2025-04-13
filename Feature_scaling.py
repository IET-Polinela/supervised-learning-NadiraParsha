  # Import library yang dibutuhkan
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Baca dataset tanpa outlier
df_no_outlier = df.copy()

# One-Hot Encoding
df_no_outlier_encoded = pd.get_dummies(df_no_outlier, drop_first=True)

# Pisahkan fitur (X) dan target (Y)
X_clean = df_no_outlier_encoded.drop('SalePrice', axis=1)

# Scaling dengan StandardScaler dan MinMaxScaler
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

# Melakukan scaling
X_scaled_standard = scaler_standard.fit_transform(X_clean)
X_scaled_minmax = scaler_minmax.fit_transform(X_clean)

# Ubah hasil scaling ke DataFrame agar mudah divisualisasikan
X_scaled_standard_df = pd.DataFrame(X_scaled_standard, columns=X_clean.columns)
X_scaled_minmax_df = pd.DataFrame(X_scaled_minmax, columns=X_clean.columns)

# Tentukan jumlah fitur
num_features = len(X_clean.columns)

# Tentukan jumlah baris dan kolom untuk grid (misalnya 4 baris dan 5 kolom)
rows = (num_features // 5) + 1  # Menentukan jumlah baris, tambah 1 untuk mengakomodasi semua fitur
cols = 5  # Tentukan jumlah kolom

# Visualisasi distribusi data sebelum dan sesudah scaling dengan histogram
plt.figure(figsize=(20, 15))

# Histogram sebelum scaling
for i, column in enumerate(X_clean.columns, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(X_clean[column], kde=True)
    plt.title(f'Sebelum Scaling: {column}')

plt.tight_layout()
plt.show()

# Histogram setelah scaling dengan StandardScaler
plt.figure(figsize=(20, 15))
for i, column in enumerate(X_clean.columns, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(X_scaled_standard_df[column], kde=True)
    plt.title(f'Setelah StandardScaler: {column}')

plt.tight_layout()
plt.show()

# Histogram setelah scaling dengan MinMaxScaler
plt.figure(figsize=(20, 15))
for i, column in enumerate(X_clean.columns, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(X_scaled_minmax_df[column], kde=True)
    plt.title(f'Setelah MinMaxScaler: {column}')

plt.tight_layout()
plt.show()
