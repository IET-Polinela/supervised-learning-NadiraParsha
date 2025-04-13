# Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor

# Baca Dataset
df = pd.read_csv('HousePricing.csv')

# Hapus kolom dengan missing value lebih dari 80%
columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
df.drop(columns=columns_to_drop, inplace=True)

# Tangani missing value numerik dengan median
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())

# Tangani missing value kategorikal dengan modus
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# One-Hot Encoding untuk fitur kategorikal
df_encoded = pd.get_dummies(df, drop_first=True)

# Pisahkan fitur dan target
X = df_encoded.drop('SalePrice', axis=1)
Y = df_encoded['SalePrice']

# Hilangkan outlier dari kolom numerik
X_numeric = X.select_dtypes(include=[np.number])
Q1 = X_numeric.quantile(0.25)
Q3 = X_numeric.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X_numeric < (Q1 - 1.5 * IQR)) | (X_numeric > (Q3 + 1.5 * IQR))).any(axis=1)
X_clean = X[mask]
Y_clean = Y[mask]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Bagi data menjadi latih dan uji
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_clean, test_size=0.2, random_state=42)

# Fungsi untuk pelatihan dan evaluasi model Polynomial Regression
def train_poly_model(degree):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, Y_train)

    Y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print(f"\nPolynomial Degree: {degree}")
    print("MSE:", mse)
    print("R² Score:", r2)

    # Visualisasi
    plt.figure(figsize=(14, 4))
    
    # Scatter plot: Prediksi vs Aktual
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=Y_test, y=Y_pred)
    plt.xlabel("Aktual")
    plt.ylabel("Prediksi")
    plt.title(f"Scatter Plot Degree {degree}")

    # Residual Plot
    plt.subplot(1, 3, 2)
    residuals = Y_test - Y_pred
    sns.scatterplot(x=Y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Prediksi")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot Degree {degree}")

    # Distribusi Residual
    plt.subplot(1, 3, 3)
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"Distribusi Residual Degree {degree}")
    
    plt.tight_layout()
    plt.show()

# Latih dan evaluasi model Polynomial Regression dengan degree 2 dan 3
train_poly_model(degree=2)
train_poly_model(degree=3)

# Fungsi untuk pelatihan dan evaluasi KNN Regression dengan nilai k yang berbeda
def train_knn_model(k_values):
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, Y_train)

        Y_pred_knn = knn.predict(X_test)

        mse = mean_squared_error(Y_test, Y_pred_knn)
        r2 = r2_score(Y_test, Y_pred_knn)

        print(f"\nKNN Regression (K={k})")
        print("MSE:", mse)
        print("R² Score:", r2)

        # Visualisasi: Prediksi vs Aktual
        plt.figure(figsize=(14, 4))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=Y_test, y=Y_pred_knn)
        plt.xlabel("Aktual")
        plt.ylabel("Prediksi")
        plt.title(f"Scatter Plot KNN (K={k})")

        # Residual Plot
        plt.subplot(1, 2, 2)
        residuals_knn = Y_test - Y_pred_knn
        sns.scatterplot(x=Y_pred_knn, y=residuals_knn)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Prediksi")
        plt.ylabel("Residual")
        plt.title(f"Residual Plot KNN (K={k})")
        
        plt.tight_layout()
        plt.show()

# Latih dan evaluasi model KNN Regression dengan nilai k = 3, 5, 7
k_values = [3, 5, 7]
train_knn_model(k_values)

# Fungsi untuk pelatihan dan evaluasi Linear Regression
def train_lr_model():
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    Y_pred_lr = lr.predict(X_test)

    mse = mean_squared_error(Y_test, Y_pred_lr)
    r2 = r2_score(Y_test, Y_pred_lr)

    print("\nLinear Regression")
    print("MSE:", mse)
    print("R² Score:", r2)

    # Visualisasi: Prediksi vs Aktual
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=Y_test, y=Y_pred_lr)
    plt.xlabel("Aktual")
    plt.ylabel("Prediksi")
    plt.title(f"Scatter Plot Linear Regression")

    # Residual Plot
    plt.subplot(1, 2, 2)
    residuals_lr = Y_test - Y_pred_lr
    sns.scatterplot(x=Y_pred_lr, y=residuals_lr)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Prediksi")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot Linear Regression")
    
    plt.tight_layout()
    plt.show()

# Latih dan evaluasi model Linear Regression
train_lr_model()

 