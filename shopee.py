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

# --- Statistik Sentimen (Data Asli) ---
st.subheader("Statistik Sentimen (Data Asli)")

fig_sentimen, ax_sentimen = plt.subplots()

sentimen_counts = df["Sentimen"].value_counts()
sentimen_counts.plot(kind="bar", ax=ax_sentimen, color="lightgreen")
# Menambahkan angka di atas bar
for i, v in enumerate(sentimen_counts):
    ax_sentimen.text(i, v + 1, str(v), ha='center', va='bottom')
st.pyplot(fig_sentimen)

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

# --- Statistik Sentimen (Data Asli) ---
st.subheader("Statistik Sentimen (Data Asli)")

fig_sentimen, ax_sentimen = plt.subplots()

sentimen_counts = df["Sentimen"].value_counts()
sentimen_counts.plot(kind="bar", ax=ax_sentimen, color="lightgreen")
# Menambahkan angka di atas bar
for i, v in enumerate(sentimen_counts):
    ax_sentimen.text(i, v + 1, str(v), ha='center', va='bottom')
st.pyplot(fig_sentimen)

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
