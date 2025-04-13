  # Import library yang dibutuhkan
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

# Visualisasi Boxplot untuk identifikasi outlier
numeric_features = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(20, 15))
for i, column in enumerate(numeric_features.columns, 1):
    plt.subplot((len(numeric_features.columns) // 3) + 1, 3, i)
    sns.boxplot(x=numeric_features[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# Penanganan outlier menggunakan metode IQR
df_no_outlier = df.copy()

for col in numeric_features.columns:
    Q1 = df_no_outlier[col].quantile(0.25)
    Q3 = df_no_outlier[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outlier = df_no_outlier[(df_no_outlier[col] >= lower_bound) & (df_no_outlier[col] <= upper_bound)]

# Encoding untuk fitur kategorikal (One-Hot Encoding)
df_encoded = pd.get_dummies(df, drop_first=True)
df_no_outlier_encoded = pd.get_dummies(df_no_outlier, drop_first=True)

# Pisahkan fitur dan target
X = df_encoded.drop('SalePrice', axis=1)
Y = df_encoded['SalePrice']

X_clean = df_no_outlier_encoded.drop('SalePrice', axis=1)
Y_clean = df_no_outlier_encoded['SalePrice']

# Bagi data menjadi training dan testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train_clean, X_test_clean, Y_train_clean, Y_test_clean = train_test_split(X_clean, Y_clean, test_size=0.2, random_state=42)

# Tampilkan hasil
print("Dataset asli (dengan outlier):")
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("Y_train:", Y_train.shape, "| Y_test:", Y_test.shape)

print("\nDataset tanpa outlier:")
print("X_train_clean:", X_train_clean.shape, "| X_test_clean:", X_test_clean.shape)
print("Y_train_clean:", Y_train_clean.shape, "| Y_test_clean:", Y_test_clean.shape)
