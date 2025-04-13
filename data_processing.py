  # Import library yang dibutuhkan
import pandas as pd
from sklearn.model_selection import train_test_split

# Baca dataset
df = pd.read_csv('HousePricing.csv')

# Hapus fitur yang memiliki missing value terlalu banyak (> 80%)
columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
df.drop(columns=columns_to_drop, inplace=True)

# Tangani missing value numerik dengan median
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())

# Tangani missing value kategorikal dengan modus
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# Lakukan One-Hot Encoding untuk fitur kategorikal
df_encoded = pd.get_dummies(df, drop_first=True)

# Pisahkan fitur (X) dan target (Y)
X = df_encoded.drop('SalePrice', axis=1)
Y = df_encoded['SalePrice']

# Bagi data menjadi data latih dan data uji (80:20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Tampilkan bentuk data untuk memastikan
print("Ukuran X_train:", X_train.shape)
print("Ukuran X_test :", X_test.shape)
print("Ukuran Y_train:", Y_train.shape)
print("Ukuran Y_test :", Y_test.shape)
