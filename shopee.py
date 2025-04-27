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
 
 # --- Simpan inputan ulasan dan hasil prediksi ---
 if "input_text" not in st.session_state:
     st.session_state.input_text = ""
 
 if "last_prediction" not in st.session_state:
     st.session_state.last_prediction = {}
 
 # Dataframe untuk simpan semua hasil prediksi berdasarkan model
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
     "Pilih Model Prediksi",
     ("CatBoost", "GradientBoosting", "Gabungan (Voting)")
 )
 
 predict_btn = st.button("Prediksi")
 
 # --- Proses Prediksi ---
 if predict_btn and input_text.strip() != "":
     st.session_state.input_text = input_text  # Simpan input
 
     input_vec = vectorizer.transform([input_text])
 
     # Prediksi masing-masing model
     pred_sentimen_cat = catboost_sentimen.predict(input_vec)[0]
     pred_sentimen_gbc = gbc_sentimen.predict(input_vec)[0]
 
     if y_aspek is not None:
         pred_aspek_cat = catboost_aspek.predict(input_vec)[0]
         pred_aspek_gbc = gbc_aspek.predict(input_vec)[0]
     else:
         pred_aspek_cat = "Unknown"
         pred_aspek_gbc = "Unknown"
 
     # Gabungan Voting
     pred_sentimen_vote = pred_sentimen_cat if pred_sentimen_cat == pred_sentimen_gbc else "Netral"
     pred_aspek_vote = pred_aspek_cat if pred_aspek_cat == pred_aspek_gbc else "Gabungan"
 
     # Simpan hasil terakhir
     if selected_model == "CatBoost":
         st.session_state.last_prediction = {
             "Ulasan": input_text,
             "Aspek": pred_aspek_cat,
             "Sentimen": pred_sentimen_cat
         }
         st.session_state.data_pred_catboost = pd.concat([
             st.session_state.data_pred_catboost,
             pd.DataFrame([st.session_state.last_prediction])
         ], ignore_index=True)
 
     elif selected_model == "GradientBoosting":
         st.session_state.last_prediction = {
             "Ulasan": input_text,
             "Aspek": pred_aspek_gbc,
             "Sentimen": pred_sentimen_gbc
         }
         st.session_state.data_pred_gbc = pd.concat([
             st.session_state.data_pred_gbc,
             pd.DataFrame([st.session_state.last_prediction])
         ], ignore_index=True)
 
     else:  # Voting
         st.session_state.last_prediction = {
             "Ulasan": input_text,
             "Aspek": pred_aspek_vote,
             "Sentimen": pred_sentimen_vote
         }
         st.session_state.data_pred_voting = pd.concat([
             st.session_state.data_pred_voting,
             pd.DataFrame([st.session_state.last_prediction])
         ], ignore_index=True)
 
 # --- Tampilkan hasil prediksi terakhir ---
 if st.session_state.last_prediction:
     st.subheader("Hasil Prediksi Terakhir")
     st.write("**Ulasan:**", st.session_state.last_prediction["Ulasan"])
     st.write("**Aspek:**", st.session_state.last_prediction["Aspek"])
     st.write("**Sentimen:**", st.session_state.last_prediction["Sentimen"])
 
 # --- Pilihan untuk lihat semua data prediksi ---
 st.subheader("Lihat Rekap Semua Prediksi")
 
 model_to_view = st.selectbox(
     "Pilih Model untuk Melihat Data",
     ("-", "CatBoost", "GradientBoosting", "Gabungan (Voting)")
 )
 
 if model_to_view != "-":
     if model_to_view == "CatBoost":
         df_view = st.session_state.data_pred_catboost
     elif model_to_view == "GradientBoosting":
         df_view = st.session_state.data_pred_gbc
     else:
         df_view = st.session_state.data_pred_voting
 
     st.dataframe(df_view)
 
     # Statistik aspek
     st.subheader(f"Statistik Aspek ({model_to_view})")
     if not df_view.empty:
         fig_aspek, ax_aspek = plt.subplots()
         df_view["Aspek"].value_counts().plot(kind="bar", ax=ax_aspek, color="skyblue")
         for i, v in enumerate(df_view["Aspek"].value_counts()):
             ax_aspek.text(i, v + 1, str(v), ha='center', va='bottom')
         st.pyplot(fig_aspek)
     else:
         st.write("Belum ada data aspek.")
 
     # Statistik sentimen
     st.subheader(f"Statistik Sentimen ({model_to_view})")
     if not df_view.empty:
         fig_sentimen, ax_sentimen = plt.subplots()
         df_view["Sentimen"].value_counts().plot(kind="bar", ax=ax_sentimen, color="lightgreen")
         for i, v in enumerate(df_view["Sentimen"].value_counts()):
             ax_sentimen.text(i, v + 1, str(v), ha='center', va='bottom')
         st.pyplot(fig_sentimen)
     else:
         st.write("Belum ada data sentimen.")
