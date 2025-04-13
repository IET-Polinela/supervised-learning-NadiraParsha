  # Import additional libraries for KNN
from sklearn.neighbors import KNeighborsRegressor

# Fungsi untuk pelatihan dan evaluasi model KNN Regression
def train_knn_model(k_values):
    for k in k_values:
        # Latih model KNN dengan nilai k tertentu
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, Y_train)
        
        # Prediksi dengan model KNN
        Y_pred_knn = knn.predict(X_test)
        
        # Hitung MSE dan R²
        mse_knn = mean_squared_error(Y_test, Y_pred_knn)
        r2_knn = r2_score(Y_test, Y_pred_knn)
        
        print(f"\nKNN (k={k})")
        print("MSE:", mse_knn)
        print("R² Score:", r2_knn)

        # Visualisasi
        plt.figure(figsize=(14, 4))

        # Scatter plot: Prediksi vs Aktual
        plt.subplot(1, 3, 1)
        sns.scatterplot(x=Y_test, y=Y_pred_knn)
        plt.xlabel("Aktual")
        plt.ylabel("Prediksi")
        plt.title(f"Scatter Plot KNN (k={k})")

        # Residual Plot
        plt.subplot(1, 3, 2)
        residuals_knn = Y_test - Y_pred_knn
        sns.scatterplot(x=Y_pred_knn, y=residuals_knn)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Prediksi")
        plt.ylabel("Residual")
        plt.title(f"Residual Plot KNN (k={k})")

        # Distribusi Residual
        plt.subplot(1, 3, 3)
        sns.histplot(residuals_knn, kde=True, bins=30)
        plt.title(f"Distribusi Residual KNN (k={k})")

        plt.tight_layout()
        plt.show()

# Tentukan nilai K untuk KNN
k_values = [3, 5, 7]

# Latih dan evaluasi model KNN
train_knn_model(k_values)

# Perbandingan antara Linear, Polynomial, dan KNN Regression
print("\n=== Perbandingan Model ===")
models = ['Linear Regression', 'Polynomial Degree 2', 'Polynomial Degree 3', 'KNN (k=3)', 'KNN (k=5)', 'KNN (k=7)']
mse_values = []
r2_values = []

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(Y_test, Y_pred_lr)
r2_lr = r2_score(Y_test, Y_pred_lr)
mse_values.append(mse_lr)
r2_values.append(r2_lr)

# Polynomial Regression degree=2
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)
lr_poly2 = LinearRegression()
lr_poly2.fit(X_train_poly2, Y_train)
Y_pred_poly2 = lr_poly2.predict(X_test_poly2)
mse_poly2 = mean_squared_error(Y_test, Y_pred_poly2)
r2_poly2 = r2_score(Y_test, Y_pred_poly2)
mse_values.append(mse_poly2)
r2_values.append(r2_poly2)

# Polynomial Regression degree=3
poly3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)
lr_poly3 = LinearRegression()
lr_poly3.fit(X_train_poly3, Y_train)
Y_pred_poly3 = lr_poly3.predict(X_test_poly3)
mse_poly3 = mean_squared_error(Y_test, Y_pred_poly3)
r2_poly3 = r2_score(Y_test, Y_pred_poly3)
mse_values.append(mse_poly3)
r2_values.append(r2_poly3)

# KNN Regression for k=3
knn_3 = KNeighborsRegressor(n_neighbors=3)
knn_3.fit(X_train, Y_train)
Y_pred_knn3 = knn_3.predict(X_test)
mse_knn3 = mean_squared_error(Y_test, Y_pred_knn3)
r2_knn3 = r2_score(Y_Test, Y_pred_knn3)
mse_values.append(mse_knn3)
r2_values.append(r2_knn3)

# KNN Regression for k=5
knn_5 = KNeighborsRegressor(n_neighbors=5)
knn_5.fit(X_train, Y_train)
Y_pred_knn5 = knn_5.predict(X_test)
mse_knn5 = mean_squared_error(Y_test, Y_pred_knn5)
r2_knn5 = r2_score(Y_Test, Y_pred_knn5)
mse_values.append(mse_knn5)
r2_values.append(r2_knn5)

# KNN Regression for k=7
knn_7 = KNeighborsRegressor(n_neighbors=7)
knn_7.fit(X_train, Y_train)
Y_pred_knn7 = knn_7.predict(X_test)
mse_knn7 = mean_squared_error(Y_Test, Y_pred_knn7)
r2_knn7 = r2_score(Y_Test, Y_pred_knn7)
mse_values.append(mse_knn7)
r2_values.append(r2_knn7)

# Visualisasi Perbandingan MSE dan R² untuk semua model
plt.figure(figsize=(14, 6))

# MSE Comparison
plt.subplot(1, 2, 1)
plt.barh(models, mse_values, color='skyblue')
plt.xlabel("MSE")
plt.title("Perbandingan MSE")

# R² Comparison
plt.subplot(1, 2, 2)
plt.barh(models, r2_values, color='salmon')
plt.xlabel("R² Score")
plt.title("Perbandingan R²")

plt.tight_layout()
plt.show()
