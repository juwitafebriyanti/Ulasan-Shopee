import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
import numpy as np

st.session_state.data_pred = st.session_state.data_pred_catboost.copy()
st.session_state.data_pred = st.session_state.data_pred.append(st.session_state.data_pred_gbc, ignore_index=True)
st.session_state.data_pred = st.session_state.data_pred.append(st.session_state.data_pred_voting, ignore_index=True)


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

import seaborn as sns

# --- Statistik Aspek (Data Asli) ---
st.subheader("Statistik Aspek (Data Asli)")

fig_aspek, ax_aspek = plt.subplots()

if 'Aspek' in df.columns:
    aspect_counts = df["Aspek"].value_counts()
    aspect_counts.plot(kind="bar", ax=ax_aspek, color="skyblue")
    # Menambahkan angka di atas bar
    for i, v in enumerate(aspect_counts):
        ax_aspek.text(i, v + 1, str(v), ha='center', va='bottom')
    st.pyplot(fig_aspek)
else:
    st.write("Data aspek tidak tersedia.")

# --- Statistik Sentimen (Data Asli) ---
st.subheader("Statistik Sentimen (Data Asli)")

fig_sentimen, ax_sentimen = plt.subplots()

sentimen_counts = df["Sentimen"].value_counts()
sentimen_counts.plot(kind="bar", ax=ax_sentimen, color="lightgreen")
# Menambahkan angka di atas bar
for i, v in enumerate(sentimen_counts):
    ax_sentimen.text(i, v + 1, str(v), ha='center', va='bottom')
st.pyplot(fig_sentimen)

# --- Simpan inputan ulasan ke session ---
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# --- Simpan hasil prediksi per model ---
if "data_pred_catboost" not in st.session_state:
    st.session_state.data_pred_catboost = pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"])

if "data_pred_gbc" not in st.session_state:
    st.session_state.data_pred_gbc = pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"])

if "data_pred_voting" not in st.session_state:
    st.session_state.data_pred_voting = pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"])

# --- Input Ulasan ---
st.subheader("Input Ulasan Baru")
input_text = st.text_area("Masukkan Ulasan", st.session_state.input_text)
selected_model = st.selectbox(
    "Pilih Model",
    ("CatBoost", "GradientBoosting", "Gabungan (Voting)")
)

predict_btn = st.button("Prediksi")

# Kalau klik prediksi atau model berganti
if predict_btn and input_text.strip() != "":
    st.session_state.input_text = input_text  # Simpan input supaya bisa pakai lagi
    input_vec = vectorizer.transform([input_text])

    # Prediksi Sentimen
    pred_sentimen_cat = catboost_sentimen.predict(input_vec)[0]
    pred_sentimen_gbc = gbc_sentimen.predict(input_vec)[0]

    # Prediksi Aspek
    if y_aspek is not None:
        pred_aspek_cat = catboost_aspek.predict(input_vec)[0]
        pred_aspek_gbc = gbc_aspek.predict(input_vec)[0]
    else:
        pred_aspek_cat = "Unknown"
        pred_aspek_gbc = "Unknown"

    # Gabungan Voting
    if pred_sentimen_cat == pred_sentimen_gbc:
        pred_sentimen_vote = pred_sentimen_cat
    else:
        pred_sentimen_vote = "Netral"

    if pred_aspek_cat == pred_aspek_gbc:
        pred_aspek_vote = pred_aspek_cat
    else:
        pred_aspek_vote = "Gabungan"

    # Simpan hasil berdasarkan model
    st.session_state.data_pred_catboost = pd.concat([
        st.session_state.data_pred_catboost,
        pd.DataFrame([{"Ulasan": input_text, "Aspek": pred_aspek_cat, "Sentimen": pred_sentimen_cat}])
    ], ignore_index=True)

    st.session_state.data_pred_gbc = pd.concat([
        st.session_state.data_pred_gbc,
        pd.DataFrame([{"Ulasan": input_text, "Aspek": pred_aspek_gbc, "Sentimen": pred_sentimen_gbc}])
    ], ignore_index=True)

    st.session_state.data_pred_voting = pd.concat([
        st.session_state.data_pred_voting,
        pd.DataFrame([{"Ulasan": input_text, "Aspek": pred_aspek_vote, "Sentimen": pred_sentimen_vote}])
    ], ignore_index=True)

# --- Tampilkan hasil sesuai model yang dipilih ---
st.subheader(f"Hasil Prediksi ({selected_model})")

if selected_model == "CatBoost":
    st.dataframe(st.session_state.data_pred_catboost)
elif selected_model == "GradientBoosting":
    st.dataframe(st.session_state.data_pred_gbc)
else:
    st.dataframe(st.session_state.data_pred_voting)


# --- Statistik Aspek (Prediksi) ---
st.subheader("Statistik Aspek (Prediksi)")

fig_aspek_pred, ax_aspek_pred = plt.subplots()

if not st.session_state.data_pred.empty and "Aspek" in st.session_state.data_pred.columns:
    aspect_pred_counts = st.session_state.data_pred["Aspek"].value_counts()
    aspect_pred_counts.plot(kind="bar", ax=ax_aspek_pred, color="skyblue")
    # Menambahkan angka di atas bar
    for i, v in enumerate(aspect_pred_counts):
        ax_aspek_pred.text(i, v + 1, str(v), ha='center', va='bottom')
    st.pyplot(fig_aspek_pred)
else:
    st.write("Belum ada data aspek untuk ditampilkan.")

# --- Statistik Sentimen (Prediksi) ---
st.subheader("Statistik Sentimen (Prediksi)")

fig_sentimen_pred, ax_sentimen_pred = plt.subplots()

if not st.session_state.data_pred.empty and "Sentimen" in st.session_state.data_pred.columns:
    sentimen_pred_counts = st.session_state.data_pred["Sentimen"].value_counts()
    sentimen_pred_counts.plot(kind="bar", ax=ax_sentimen_pred, color="lightgreen")
    # Menambahkan angka di atas bar
    for i, v in enumerate(sentimen_pred_counts):
        ax_sentimen_pred.text(i, v + 1, str(v), ha='center', va='bottom')
    st.pyplot(fig_sentimen_pred)
else:
    st.write("Belum ada data sentimen untuk ditampilkan.")

