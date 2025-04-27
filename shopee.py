import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from catboost import CatBoostClassifier
import numpy as np

# --- Load Dataset ---
df = pd.read_excel("Dataset/ulasan_shopee_preprocessed.xlsx")  # Pastikan path benar

# # --- Load Data ---
# X_raw = df['Ulasan_Clean']
# y_sentimen = df['Sentimen']

# # Kalau ada kolom Aspek, misal Aspek Ulasan
# if 'Aspek' in df.columns:
#     y_aspek = df['Aspek']
# else:
#     y_aspek = None

# # --- TF-IDF ---
# vectorizer = TfidfVectorizer(max_features=5000)
# X = vectorizer.fit_transform(X_raw)

# # --- Split ---
# X_train, X_test, y_train_sentimen, y_test_sentimen = train_test_split(X, y_sentimen, test_size=0.2, random_state=42)

# # --- Train Models ---
# gbc_sentimen = GradientBoostingClassifier(random_state=42)
# gbc_sentimen.fit(X_train, y_train_sentimen)

# catboost_sentimen = CatBoostClassifier(verbose=0, random_state=42)
# catboost_sentimen.fit(X_train, y_train_sentimen)

# # Kalau mau aspek juga diprediksi, perlu training model aspek juga
# if y_aspek is not None:
#     y_train_aspek, y_test_aspek = train_test_split(y_aspek, test_size=0.2, random_state=42)
#     gbc_aspek = GradientBoostingClassifier(random_state=42)
#     gbc_aspek.fit(X_train, y_train_aspek)
#     catboost_aspek = CatBoostClassifier(verbose=0, random_state=42)
#     catboost_aspek.fit(X_train, y_train_aspek)

# --- Streamlit App ---

st.title("Analisis Kepuasan Pengguna Shopee 🛒")
st.write("Prediksi Aspek dan Sentimen Ulasan Shopee")

# Input
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
    st.session_state.data_pred = pd.concat([
        st.session_state.data_pred,
        pd.DataFrame([{"Ulasan": input_text, "Aspek": final_aspek, "Sentimen": final_sentimen}])
    ], ignore_index=True)

# --- Display Result ---
st.subheader("Hasil Prediksi")

st.dataframe(st.session_state.data_pred)

# --- Statistik ---
st.subheader("Statistik Aspek")
fig_aspek, ax_aspek = plt.subplots()
st.session_state.data_pred["Aspek"].value_counts().plot(kind="bar", ax=ax_aspek, color="skyblue")
st.pyplot(fig_aspek)

st.subheader("Statistik Sentimen")
fig_sentimen, ax_sentimen = plt.subplots()
st.session_state.data_pred["Sentimen"].value_counts().plot(kind="bar", ax=ax_sentimen, color="lightgreen")
st.pyplot(fig_sentimen)
