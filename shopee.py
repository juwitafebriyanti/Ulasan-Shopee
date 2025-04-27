import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier

# --- Load Dataset ---
df = pd.read_excel("Dataset/ulasan_shopee_preprocessed.xlsx")  # Pastikan path benar

# --- Load Data ---
X_raw = df['Ulasan_Clean']
y_sentimen = df['Sentimen']

# Kalau ada kolom Aspek
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

catboost_sentimen = CatBoostClassifier(verbose=0, random_state=42)
catboost_sentimen.fit(X_train, y_train_sentimen)

# Kalau aspek juga mau diprediksi
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
if 'lihat_selengkapnya' not in st.session_state:
    st.session_state.lihat_selengkapnya = False

if st.session_state.lihat_selengkapnya:
    st.dataframe(df)
else:
    st.dataframe(df.head(10))

if st.button("Lihat Selengkapnya"):
    st.session_state.lihat_selengkapnya = True

# --- Statistik Data Asli ---
st.subheader("Statistik Aspek (Data Asli)")
fig_aspek, ax_aspek = plt.subplots()
if 'Aspek' in df.columns:
    aspect_counts = df['Aspek'].value_counts()
    aspect_counts.plot(kind="bar", ax=ax_aspek, color="skyblue")
    for i, v in enumerate(aspect_counts):
        ax_aspek.text(i, v + 1, str(v), ha='center', va='bottom')
    st.pyplot(fig_aspek)
else:
    st.write("Data aspek tidak tersedia.")

st.subheader("Statistik Sentimen (Data Asli)")
fig_sentimen, ax_sentimen = plt.subplots()
sentimen_counts = df['Sentimen'].value_counts()
sentimen_counts.plot(kind="bar", ax=ax_sentimen, color="lightgreen")
for i, v in enumerate(sentimen_counts):
    ax_sentimen.text(i, v + 1, str(v), ha='center', va='bottom')
st.pyplot(fig_sentimen)

# --- Input Ulasan Baru ---
st.subheader("Input Ulasan Baru")

input_text = st.text_area("Masukkan Ulasan", "")

# Menyimpan input ulasan ke session state
if "input_text_saved" not in st.session_state:
    st.session_state.input_text_saved = ""

if st.button("Simpan Ulasan"):
    if input_text.strip() != "":
        st.session_state.input_text_saved = input_text
    else:
        st.warning("Masukkan ulasan terlebih dahulu!")

# Kalau sudah ada input tersimpan, tampilkan
if st.session_state.input_text_saved != "":
    st.success(f"Ulasan disimpan: {st.session_state.input_text_saved}")

    selected_model = st.selectbox(
        "Pilih Model untuk Prediksi",
        ("CatBoost", "GradientBoosting", "Gabungan (Voting)")
    )

    predict_btn = st.button("Prediksi dengan Model Terpilih")

    # Session untuk menyimpan hasil prediksi per model
    if "data_pred_per_model" not in st.session_state:
        st.session_state.data_pred_per_model = {
            "CatBoost": pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"]),
            "GradientBoosting": pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"]),
            "Gabungan (Voting)": pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"]),
        }

    if predict_btn:
        input_vec = vectorizer.transform([st.session_state.input_text_saved])

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

        # Voting Model
        if selected_model == "CatBoost":
            final_sentimen = pred_sentimen_cat
            final_aspek = pred_aspek_cat
        elif selected_model == "GradientBoosting":
            final_sentimen = pred_sentimen_gbc
            final_aspek = pred_aspek_gbc
        else:  # Voting
            final_sentimen = pred_sentimen_cat if pred_sentimen_cat == pred_sentimen_gbc else "Netral"
            final_aspek = pred_aspek_cat if pred_aspek_cat == pred_aspek_gbc else "Gabungan"

        # Simpan hasil ke dataframe model tertentu
        st.session_state.data_pred_per_model[selected_model] = pd.concat([
            st.session_state.data_pred_per_model[selected_model],
            pd.DataFrame([{
                "Ulasan": st.session_state.input_text_saved,
                "Aspek": final_aspek,
                "Sentimen": final_sentimen
            }])
        ], ignore_index=True)

# --- Display Results ---
st.subheader("Hasil Prediksi per Model")

for model_name, df_model in st.session_state.data_pred_per_model.items():
    st.write(f"### {model_name}")
    if not df_model.empty:
        st.dataframe(df_model)
    else:
        st.write("Belum ada prediksi untuk model ini.")

# --- Statistik Prediksi ---
st.subheader("Statistik Aspek dan Sentimen (Prediksi) per Model")

for model_name, df_model in st.session_state.data_pred_per_model.items():
    if not df_model.empty:
        st.subheader(f"Statistik Aspek - {model_name}")
        fig_aspek_pred, ax_aspek_pred = plt.subplots()
        aspect_pred_counts = df_model['Aspek'].value_counts()
        aspect_pred_counts.plot(kind="bar", ax=ax_aspek_pred, color="skyblue")
        for i, v in enumerate(aspect_pred_counts):
            ax_aspek_pred.text(i, v + 1, str(v), ha='center', va='bottom')
        st.pyplot(fig_aspek_pred)

        st.subheader(f"Statistik Sentimen - {model_name}")
        fig_sentimen_pred, ax_sentimen_pred = plt.subplots()
        sentimen_pred_counts = df_model['Sentimen'].value_counts()
        sentimen_pred_counts.plot(kind="bar", ax=ax_sentimen_pred, color="lightgreen")
        for i, v in enumerate(sentimen_pred_counts):
            ax_sentimen_pred.text(i, v + 1, str(v), ha='center', va='bottom')
        st.pyplot(fig_sentimen_pred)
