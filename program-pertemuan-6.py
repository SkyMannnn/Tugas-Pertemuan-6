import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Membaca dataset dan menghapus baris rekapitulasi Grand Total di bagian bawah
df = pd.read_excel(
    "persentase-desa-yang-mendapatkan-layanan-kesehatan_2.csv.xlsx"
)
df = df.dropna(subset=["KABUPATEN"]).copy()

# Membuat kolom Target: 1 jika % Desa Fasilitas Kesehatan > 0.30, selain itu 0
df["Target"] = (df["% DESA MEMILIKI FASILITAS KESEHATAN"] > 0.30).astype(int)
fitur = ["JUMLAH DESA MEMILIKI FASILITAS KESEHATAN", "TOTAL DESA"]

# ==============================================================================
# PROGRAM 1: MENGGUNAKAN 11 DATA PERTAMA (10 LATIH, 1 UJI, 5 POHON 1 FITUR)
# ==============================================================================
X_train_11 = df.iloc[:10][fitur]
y_train_11 = df.iloc[:10]["Target"]
X_test_11 = df.iloc[10:11][fitur]
y_test_11 = df.iloc[10:11]["Target"]

rf_11 = RandomForestClassifier(
    n_estimators=5, max_depth=1, max_features=1, random_state=42
)
rf_11.fit(X_train_11, y_train_11)
pred_11 = rf_11.predict(X_test_11)

print("=== HASIL PROGRAM 1: 11 DATA (10 LATIH, 1 UJI) ===")
for i, tree in enumerate(rf_11.estimators_):
  f_idx = tree.tree_.feature[0]
  thresh = tree.tree_.threshold[0]
  p_tree = int(tree.predict(X_test_11.values)[0])
  print(f"Pohon {i+1}: {fitur[f_idx]} <= {thresh:.2f} -> Prediksi = {p_tree}")
print(f"Target Asli: {y_test_11.values[0]} | Prediksi Voting RF: {pred_11[0]}")

# ==============================================================================
# PROGRAM 2: MENGGUNAKAN SELURUH DATA (430 KABUPATEN/KOTA, SPLIT 80% : 20%)
# ==============================================================================
X_all = df[fitur]
y_all = df["Target"]
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42
)

rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train_all, y_train_all)
pred_all = rf_all.predict(X_test_all)

print("\n=== HASIL PROGRAM 2: SELURUH DATA (430 DATA VALID) ===")
print(f"Data Latih: {len(X_train_all)} | Data Uji: {len(X_test_all)}")
print(f"Akurasi Model: {accuracy_score(y_test_all, pred_all) * 100:.2f}%")
print(classification_report(y_test_all, pred_all))