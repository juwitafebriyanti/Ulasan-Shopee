import streamlit as st
import pandas as pd
import re
import string
import emoji

# --- Load Data ---
df = pd.read_excel("ulasan_shopee_preprocessed.xlsx")

# List Kata Kunci Aspek
aspek_harga = ['murah', 'mahal', 'ongkir', 'diskon', 'promo', 'cashback']
aspek_pelayanan = ['cs', 'customer service', 'komplain', 'balasan', 'pengaduan', 'pelayanan']
aspek_aplikasi = ['error', 'lemot', 'fitur', 'update', 'login', 'sistem', 'bug']

# Function Klasifikasi Multi-Aspek
def aspek_klasifikasi(ulasan):
    ulasan = ulasan.lower()  # Ubah jadi huruf kecil semua
    aspek_list = []

    if any(re.search(r'\b' + word + r'\b', ulasan) for word in aspek_harga):
        aspek_list.append('Harga')
    if any(re.search(r'\b' + word + r'\b', ulasan) for word in aspek_pelayanan):
        aspek_list.append('Pelayanan')
    if any(re.search(r'\b' + word + r'\b', ulasan) for word in aspek_aplikasi):
        aspek_list.append('Aplikasi')

    return ', '.join(aspek_list) if aspek_list else None  # Jika kosong, return None

# Daftar Kata Positif dan Negatif untuk Sentimen
kata_positif = ['bagus', 'puas', 'mantap', 'baik', 'cepat', 'murah', 'ramah', 'oke', 'senang']
kata_negatif = ['buruk', 'lambat', 'jelek', 'tidak puas', 'parah', 'mahal', 'kecewa', 'lemot', 'error']

# Function Analisis Sentimen
def analisis_sentimen(ulasan):
    ulasan = str(ulasan).lower()
    skor = 0

    for kata in kata_positif:
        if kata in ulasan:
            skor += 1
    for kata in kata_negatif:
        if kata in ulasan:
            skor -= 1

    if skor > 0:
        return 'Positif'
    elif skor < 0:
        return 'Negatif'
    else:
        return 'Netral'

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()  # Ubah ke huruf kecil
    text = re.sub(r'http\S+', '', text)  # Hapus link
    text = re.sub(r'@\w+|\#', '', text)  # Hapus mention dan hashtag
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    text = emoji.replace_emoji(text, replace='')  # Hapus emoji
    text = re.sub(r'[ᗒᗣᗕ՞]', '', text)  # Hapus karakter khusus
    text = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼²³⁴⁵⁶⁷⁸⁹]', '', text)  # Hapus tanda superscript/subscript
    text = text.strip()  # Hapus spasi berlebih
    return text

# --- Streamlit App ---
st.title("Analisis Kepuasan Pengguna Shopee 🛒")
st.write("Prediksi Aspek dan Sentimen Ulasan Shopee")

# Input Ulasan
st.subheader("Masukkan Ulasan Baru")

input_text = st.text_area("Masukkan Ulasan", "")

predict_btn = st.button("Prediksi")

# Simpan ulasan + prediksi
if "data_pred" not in st.session_state:
    st.session_state.data_pred = pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"])

# Kalau klik prediksi
if predict_btn and input_text.strip() != "":
    # Preprocessing
    cleaned_text = clean_text(input_text)

    # Klasifikasi Aspek
    aspek_pred = aspek_klasifikasi(cleaned_text)

    # Sentimen
    sentimen_pred = analisis_sentimen(cleaned_text)

    # Tambahkan hasil prediksi ke dataframe
    st.session_state.data_pred = pd.concat([st.session_state.data_pred, pd.DataFrame([{"Ulasan": input_text, "Aspek": aspek_pred, "Sentimen": sentimen_pred}])], ignore_index=True)

# --- Display Hasil Prediksi ---
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
