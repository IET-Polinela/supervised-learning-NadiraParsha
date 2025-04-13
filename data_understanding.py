  
# Import pustaka
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Baca dataset
df = pd.read_csv('HousePricing.csv')

# Tampilkan 5 data pertama
print("📌 5 Data Teratas:")
display(df.head())

# Informasi struktur dataset
print("\n📌 Informasi Dataset:")
df.info()

# Statistik deskriptif (mean, std, min, Q1, median, Q3, max)
print("\n📌 Statistik Deskriptif:")
display(df.describe())

# Median (Q2) secara eksplisit
print("\n📌 Median (Q2) untuk setiap kolom:")
display(df.median(numeric_only=True))

# Jumlah data (non-null) per kolom
print("\n📌 Jumlah Data per Kolom:")
display(df.count())

# Visualisasi distribusi data numerik
print("\n📌 Visualisasi Boxplot (untuk fitur numerik):")
plt.figure(figsize=(14, 6))
sns.boxplot(data=df.select_dtypes(include=[np.number]), orient="h")
plt.title("Boxplot untuk fitur numerik (Q1, Q2, Q3)")
plt.xlabel("Nilai")
plt.grid(True)
plt.tight_layout()
plt.show()
