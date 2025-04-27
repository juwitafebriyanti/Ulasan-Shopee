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

import seaborn as sns

# --- Statistik Aspek dan Sentimen (Data Asli) ---
st.subheader("Statistik Aspek dan Sentimen (Data Asli)")

fig_aspek_sentimen_asli, ax = plt.subplots(figsize=(12,6))

if 'Aspek' in df.columns and 'Sentimen' in df.columns:
    # Hitung jumlah ulasan per Aspek dan Sentimen
    grouped = df.groupby(['Aspek', 'Sentimen']).size().reset_index(name='Jumlah')

    # Plot dengan seaborn
    sns.barplot(
        data=grouped,
        x='Aspek',
        y='Jumlah',
        hue='Sentimen',
        palette={"Positif":"mediumseagreen", "Netral":"sandybrown", "Negatif":"cornflowerblue"},
        ax=ax
    )

    # Tambahin label angka di atas bar
    for c in ax.containers:
        ax.bar_label(c, label_type='edge', fontsize=8)

    ax.set_title("Distribusi Sentimen per Aspek (Data Asli)")
    ax.set_xlabel("Aspek")
    ax.set_ylabel("Jumlah Ulasan")
    ax.legend(title="Sentimen")
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig_aspek_sentimen_asli)
else:
    st.write("Data aspek dan sentimen tidak tersedia.")



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

# --- Statistik Distribusi Sentimen per Aspek (Prediksi) ---
# --- Statistik Aspek dan Sentimen (Prediksi) ---
st.subheader("Statistik Aspek dan Sentimen (Prediksi)")

fig_aspek_sentimen_pred, ax_aspek_sentimen_pred = plt.subplots(figsize=(12,6))

if not st.session_state.data_pred.empty and "Aspek" in st.session_state.data_pred.columns and "Sentimen" in st.session_state.data_pred.columns:
    count_data_pred = st.session_state.data_pred.groupby(['Aspek', 'Sentimen']).size().unstack(fill_value=0)

    aspek_pred = count_data_pred.index.tolist()
    sentimen_pred = ['Positif', 'Netral', 'Negatif']
    colors_pred = ['#2ecc71', '#f39c12', '#9b59b6']  # hijau, oranye, ungu

    x_pred = np.arange(len(aspek_pred))
    width_pred = 0.25

    for idx, s in enumerate(sentimen_pred):
        ax_aspek_sentimen_pred.bar(x_pred + idx*width_pred, count_data_pred.get(s, [0]*len(aspek_pred)), width_pred, label=s, color=colors_pred[idx])

        # Kasih label angka di atas bar
        for i in range(len(aspek_pred)):
            ax_aspek_sentimen_pred.text(x_pred[i] + idx*width_pred, count_data_pred.get(s, [0]*len(aspek_pred))[i] + 0.5,
                                        str(count_data_pred.get(s, [0]*len(aspek_pred))[i]),
                                        ha='center', va='bottom', fontsize=8)

    ax_aspek_sentimen_pred.set_xlabel('Aspek')
    ax_aspek_sentimen_pred.set_ylabel('Jumlah Ulasan')
    ax_aspek_sentimen_pred.set_title('Distribusi Aspek dan Sentimen (Prediksi)')
    ax_aspek_sentimen_pred.set_xticks(x_pred + width_pred)
    ax_aspek_sentimen_pred.set_xticklabels(aspek_pred, rotation=45, ha='right')
    ax_aspek_sentimen_pred.legend(title='Sentimen')
    ax_aspek_sentimen_pred.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig_aspek_sentimen_pred)

else:
    st.write("Belum ada data aspek dan sentimen untuk ditampilkan.")
