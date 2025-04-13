  # Import library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Misalkan df_no_outlier_encoded adalah dataset tanpa outlier yang sudah di-encode

# Pisahkan dataset menjadi fitur dan target
X_clean = df_no_outlier_encoded.drop('SalePrice', axis=1)
Y_clean = df_no_outlier_encoded['SalePrice']

# Bagi data menjadi data latih dan data uji (80:20)
X_train_clean, X_test_clean, Y_train_clean, Y_test_clean = train_test_split(X_clean, Y_clean, test_size=0.2, random_state=42)

# Model Linear Regression
model_clean = LinearRegression()
model_clean.fit(X_train_clean, Y_train_clean)

# Prediksi menggunakan model yang dilatih pada dataset tanpa outlier
Y_pred_clean = model_clean.predict(X_test_clean)

# Menghitung MSE dan R2 untuk dataset tanpa outlier
mse_clean = mean_squared_error(Y_test_clean, Y_pred_clean)
r2_clean = r2_score(Y_test_clean, Y_pred_clean)

print("MSE (Tanpa Outlier):", mse_clean)
print("R2 Score (Tanpa Outlier):", r2_clean)

# 1. Visualisasi hasil model tanpa outlier

# Scatter plot antara nilai aktual dan prediksi
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(Y_test_clean, Y_pred_clean)
plt.plot([min(Y_test_clean), max(Y_test_clean)], [min(Y_test_clean), max(Y_test_clean)], color='red', linestyle='--')
plt.title('Scatter Plot: Aktual vs Prediksi (Tanpa Outlier)')
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')

# Residual Plot - perbaikan pada pemanggilan sns.residplot
plt.subplot(1, 2, 2)
sns.residplot(x=Y_pred_clean, y=Y_test_clean - Y_pred_clean, lowess=True, color='blue', line_kws={'color': 'red'})
plt.title('Residual Plot (Tanpa Outlier)')
plt.xlabel('Prediksi')
plt.ylabel('Residual')
plt.tight_layout()
plt.show()

# Distribusi residual
plt.figure(figsize=(8, 6))
sns.histplot(Y_test_clean - Y_pred_clean, kde=True, color='green')
plt.title('Distribusi Residual (Tanpa Outlier)')
plt.xlabel('Residual')
plt.ylabel('Frekuensi')
plt.show()
