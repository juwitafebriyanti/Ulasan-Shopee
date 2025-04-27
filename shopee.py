import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Judul Aplikasi
st.title('Analisis Kepuasan Pengguna Shopee')

st.write("""
    Aplikasi ini menganalisis kepuasan pengguna terhadap platform Shopee di Indonesia
    menggunakan Multi-Aspect Sentiment Analysis dan Machine Learning (GBC dan CatBoost).
""")

# Memuat Model GBC dan CatBoost
try:
    gbc_model = load('model_gbc.pkl')  # Model GBC untuk klasifikasi sentimen
    catboost_model = load('model_catboost.pkl')  # Model CatBoost untuk klasifikasi aspek
    le = load('label_encoder.pkl')  # Label Encoder untuk konversi label ke angka
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")

# Fungsi untuk klasifikasi sentimen
def classify_sentiment(text):
    # Menggunakan model GBC untuk analisis sentimen
    sentiment = gbc_model.predict([text])
    return sentiment[0]

# Fungsi untuk klasifikasi aspek
def classify_aspect(text):
    # Menggunakan model CatBoost untuk klasifikasi aspek
    aspect = catboost_model.predict([text])
    return aspect[0]

# Mengunggah file CSV yang berisi ulasan pengguna
file_path = st.file_uploader("Unggah File CSV dengan Ulasan Pengguna Shopee", type=["csv"])

if file_path is not None:
    try:
        df = pd.read_csv(file_path)
        
        if 'Review' in df.columns:
            # Analisis Sentimen dan Aspek
            df['Sentiment'] = df['Review'].apply(classify_sentiment)
            df['Aspect'] = df['Review'].apply(classify_aspect)

            # Menampilkan DataFrame yang telah dianalisis
            st.subheader('Hasil Analisis')
            st.dataframe(df[['Review', 'Sentiment', 'Aspect']])

            # Visualisasi Sentimen
            sentiment_counts = df['Sentiment'].value_counts()
            st.subheader('Distribusi Sentimen')
            plt.figure(figsize=(6, 4))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
            plt.xlabel('Sentimen')
            plt.ylabel('Jumlah Ulasan')
            st.pyplot()

            # Visualisasi Aspek
            aspect_counts = df['Aspect'].value_counts()
            st.subheader('Distribusi Aspek')
            plt.figure(figsize=(6, 4))
            sns.barplot(x=aspect_counts.index, y=aspect_counts.values, palette='coolwarm')
            plt.xlabel('Aspek')
            plt.ylabel('Jumlah Ulasan')
            st.pyplot()

            # Tampilkan Pie Chart untuk Sentimen
            st.subheader('Pie Chart Sentimen')
            sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis', figsize=(6, 6))
            st.pyplot()

            # Tampilkan Pie Chart untuk Aspek
            st.subheader('Pie Chart Aspek')
            aspect_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='coolwarm', figsize=(6, 6))
            st.pyplot()

            # Mengunduh Data yang sudah dianalisis
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data Hasil Analisis",
                data=csv,
                file_name='analisis_shopee.csv',
                mime='text/csv'
            )
        else:
            st.error("Kolom 'Review' tidak ditemukan di dataset.")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
