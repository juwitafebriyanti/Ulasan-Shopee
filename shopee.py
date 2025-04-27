import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
import numpy as np

# --- Load Dataset ---
df = pd.read_excel("Dataset/ulasan_shopee_preprocessed.xlsx")  # Pastikan path benar

# --- Load Data ---
X_raw = df['Ulasan_Clean']
y_sentimen = df['Sentimen']

# Kalau ada kolom Aspek, misal Aspek Ulasan
if 'Aspek' in df.columns:
    y_aspek = df['Aspek']
else:
    y_aspek = None

# --- TF-IDF ---
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_raw)

# --- Split Data ---
X_train, X_test, y_train_sentimen, y_test_sentimen = train_test_split(X, y_sentimen, test_size=0.2, random_state=42)

# --- Train Models ---
gbc_sentimen = GradientBoostingClassifier(random_state=42)
gbc_sentimen.fit(X_train, y_train_sentimen)

# CatBoost model
catboost_sentimen = CatBoostClassifier(verbose=0, random_state=42)
catboost_sentimen.fit(X_train, y_train_sentimen)

# Kalau mau aspek juga diprediksi, perlu training model aspek juga
if y_aspek is not None:
    y_train_aspek, y_test_aspek = train_test_split(y_aspek, test_size=0.2, random_state=42)
    gbc_aspek = GradientBoostingClassifier(random_state=42)
    gbc_aspek.fit(X_train, y_train_aspek)
    catboost_aspek = CatBoostClassifier(verbose=0, random_state=42)
    catboost_aspek.fit(X_train, y_train_aspek)

# --- Streamlit App ---
st.title("Analisis Kepuasan Pengguna Shopee 🛒")
st.write("Prediksi Aspek dan Sentimen Ulasan Shopee")

# --- Display Data Ulasan ---
st.subheader("Data Ulasan Shopee")
# Menampilkan 10 data pertama
if 'lihat_selengkapnya' not in st.session_state:
    st.session_state.lihat_selengkapnya = False

if st.session_state.lihat_selengkapnya:
    # Menampilkan seluruh data jika tombol sudah diklik
    st.dataframe(df)
else:
    # Menampilkan hanya 10 data pertama
    st.dataframe(df.head(10))

# Tombol untuk melihat data selengkapnya
if st.button("Lihat Selengkapnya"):
    st.session_state.lihat_selengkapnya = True

# --- Statistik Distribusi Aspek dan Sentimen (Data Asli) ---
st.subheader("Statistik Aspek dan Sentimen")

# Menyiapkan data statistik untuk masing-masing aspek dan sentimen
if 'Aspek' in df.columns:
    aspek_sentimen_counts = df.groupby('Aspek')['Sentimen'].value_counts().unstack().fillna(0)
    
    # Membuat bar plot dengan 3 kategori sentimen untuk setiap aspek
    fig, ax = plt.subplots(figsize=(12, 8))
    aspek_sentimen_counts.plot(kind='bar', stacked=False, ax=ax, color=["lightgreen", "skyblue", "lightcoral"])

    ax.set_title("Statistik Sentimen Berdasarkan Aspek (Data Asli)")
    ax.set_ylabel("Jumlah")
    ax.set_xlabel("Aspek")
    ax.legend(title="Sentimen", labels=["Positif", "Netral", "Negatif"])

    # Menambahkan angka di atas setiap bar
    for i, aspect in enumerate(aspek_sentimen_counts.index):
        for j, sentiment in enumerate(aspek_sentimen_counts.columns):
            value = aspek_sentimen_counts.iloc[i, j]
            ax.text(i, value + 0.5, str(int(value)), ha='center', va='bottom')

    st.pyplot(fig)
    
# --- Input Ulasan Baru ---
st.subheader("Input Ulasan Baru")

input_text = st.text_area("Masukkan Ulasan", "")
selected_model = st.selectbox(
    "Pilih Model",
    ("CatBoost", "GradientBoosting", "Gabungan (Voting)")
)

predict_btn = st.button("Prediksi")

# Simpan ulasan + prediksi
if "data_pred" not in st.session_state:
    st.session_state.data_pred = pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"])

# Kalau klik prediksi
if predict_btn and input_text.strip() != "":
    # Preprocess input
    input_vec = vectorizer.transform([input_text])

    # Sentimen
    pred_sentimen_cat = catboost_sentimen.predict(input_vec)[0]
    pred_sentimen_gbc = gbc_sentimen.predict(input_vec)[0]

    # Aspek (jika ada)
    if y_aspek is not None:
        pred_aspek_cat = catboost_aspek.predict(input_vec)[0]
        pred_aspek_gbc = gbc_aspek.predict(input_vec)[0]
    else:
        pred_aspek_cat = "Unknown"
        pred_aspek_gbc = "Unknown"

    # Gabungan Voting
    if selected_model == "CatBoost":
        final_sentimen = pred_sentimen_cat
        final_aspek = pred_aspek_cat
    elif selected_model == "GradientBoosting":
        final_sentimen = pred_sentimen_gbc
        final_aspek = pred_aspek_gbc
    else:  # Voting
        final_sentimen = pred_sentimen_cat if pred_sentimen_cat == pred_sentimen_gbc else "Netral"
        final_aspek = pred_aspek_cat if pred_aspek_cat == pred_aspek_gbc else "Gabungan"

    # Tambahkan ke dataframe session
    st.session_state.data_pred = pd.concat([st.session_state.data_pred, pd.DataFrame([{"Ulasan": input_text, "Aspek": final_aspek, "Sentimen": final_sentimen}])], ignore_index=True)

# --- Display Result ---
st.subheader("Hasil Prediksi")

st.dataframe(st.session_state.data_pred)

st.subheader("Statistik Aspek dan Sentimen (Prediksi)")

# Gabungkan statistik Aspek dan Sentimen Prediksi dalam satu plot
if 'Aspek' in st.session_state.data_pred.columns:
        pred_aspek_sentimen_counts = st.session_state.data_pred.groupby('Aspek')['Sentimen'].value_counts().unstack().fillna(0)

        # Membuat bar plot untuk data prediksi
        fig_pred, ax_pred = plt.subplots(figsize=(12, 8))
        pred_aspek_sentimen_counts.plot(kind='bar', stacked=False, ax=ax_pred, color=["lightgreen", "skyblue", "lightcoral"])

        ax_pred.set_title("Statistik Sentimen Berdasarkan Aspek (Data Prediksi)")
        ax_pred.set_ylabel("Jumlah")
        ax_pred.set_xlabel("Aspek")
        ax_pred.legend(title="Sentimen", labels=["Positif", "Netral", "Negatif"])

        # Menambahkan angka di atas setiap bar
        for i, aspect in enumerate(pred_aspek_sentimen_counts.index):
            for j, sentiment in enumerate(pred_aspek_sentimen_counts.columns):
                value = pred_aspek_sentimen_counts.iloc[i, j]
                ax_pred.text(i, value + 0.5, str(int(value)), ha='center', va='bottom')

        st.pyplot(fig_pred)
else:
    st.write("Tidak ada data aspek untuk ditampilkan.")
