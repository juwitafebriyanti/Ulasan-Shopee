import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier

# --- Keyword Aspek ---
aspect_keywords = {
    "harga": ['murah', 'mahal', 'ongkir', 'diskon', 'promo', 'cashback'],
    "pelayanan": ['cs', 'customer service', 'komplain', 'balasan', 'pengaduan', 'pelayanan'],
    "aplikasi": ['error', 'lemot', 'fitur', 'update', 'login', 'sistem', 'bug'],
}

# --- Keyword Sentimen ---
kata_positif = ['bagus', 'puas', 'mantap', 'baik', 'cepat', 'murah', 'ramah', 'oke', 'senang']
kata_negatif = ['buruk', 'lambat', 'jelek', 'tidak puas', 'parah', 'mahal', 'kecewa', 'lemot', 'error']

# --- Fungsi Rule-Based ---
def detect_aspects_rule_based(text):
    aspects_found = set()
    text_lower = text.lower()
    for aspect, keywords in aspect_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                aspects_found.add(aspect)
                break
    return list(aspects_found) if aspects_found else ["Unknown"]

def detect_sentiment_rule_based(text):
    text_lower = text.lower()
    positif = any(kw in text_lower for kw in kata_positif)
    negatif = any(kw in text_lower for kw in kata_negatif)

    if positif and not negatif:
        return "Positif"
    elif negatif and not positif:
        return "Negatif"
    elif positif and negatif:
        return "Netral"
    else:
        return "Netral"

# --- Load Dataset ---
df = pd.read_excel("Dataset/ulasan_shopee_preprocessed.xlsx")
X_raw = df['Ulasan_Clean']
y_sentimen = df['Sentimen']

if 'Aspek' in df.columns:
    y_aspek = df['Aspek']
else:
    y_aspek = None

# --- TF-IDF ---
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_raw)

# --- Split Data ---
X_train, X_test, y_train_sentimen, y_test_sentimen = train_test_split(X, y_sentimen, test_size=0.2, random_state=42)

if y_aspek is not None:
    _, _, y_train_aspek, y_test_aspek = train_test_split(X, y_aspek, test_size=0.2, random_state=42)
else:
    y_train_aspek = y_test_aspek = None

# --- Train Models ---
@st.cache_resource
def train_models(_X, y_sentimen, y_aspek=None):
    gbc_sentimen = GradientBoostingClassifier(random_state=42)
    gbc_sentimen.fit(_X, y_sentimen)

    catboost_sentimen = CatBoostClassifier(verbose=0, random_state=42)
    catboost_sentimen.fit(_X.toarray(), y_sentimen)

    if y_aspek is not None:
        gbc_aspek = GradientBoostingClassifier(random_state=42)
        gbc_aspek.fit(_X, y_aspek)

        catboost_aspek = CatBoostClassifier(verbose=0, random_state=42)
        catboost_aspek.fit(_X.toarray(), y_aspek)

        return gbc_sentimen, catboost_sentimen, gbc_aspek, catboost_aspek

    return gbc_sentimen, catboost_sentimen, None, None

gbc_sentimen, catboost_sentimen, gbc_aspek, catboost_aspek = train_models(X_train, y_train_sentimen, y_train_aspek)

# --- Streamlit App ---
st.markdown("<h1 style='text-align: center;'>Analisis Kepuasan Pengguna Shopee</h1>", unsafe_allow_html=True)
st.write("Prediksi Aspek dan Sentimen Ulasan Shopee")

# --- Display Data Ulasan ---
st.subheader("Data Ulasan Shopee")
df_display = df[['Ulasan_Clean', 'Aspek', 'Sentimen']] if 'Aspek' in df.columns else df[['Ulasan_Clean', 'Sentimen']]

if 'lihat_selengkapnya' not in st.session_state:
    st.session_state.lihat_selengkapnya = False

if st.session_state.lihat_selengkapnya:
    show_df = df_display
else:
    show_df = df_display.head(10)

st.dataframe(show_df.style.set_properties(**{'white-space': 'pre-wrap','word-wrap': 'break-word'}), hide_index=True, use_container_width=True)

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

if "input_text_saved" not in st.session_state:
    st.session_state.input_text_saved = ""

if st.button("Simpan Ulasan"):
    if input_text.strip() != "":
        st.session_state.input_text_saved = input_text
    else:
        st.warning("Masukkan ulasan terlebih dahulu!")

if st.session_state.input_text_saved != "":
    st.success(f"Ulasan disimpan: {st.session_state.input_text_saved}")

    selected_model = st.selectbox("Pilih Model untuk Prediksi", ("GradientBoosting", "CatBoost", "Gabungan (Voting)"))

    predict_btn = st.button("Prediksi dengan Model Terpilih")

    if "data_pred_per_model" not in st.session_state:
        st.session_state.data_pred_per_model = {
            "GradientBoosting": pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"]),
            "CatBoost": pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"]),
            "Gabungan (Voting)": pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"]),
        }

    if predict_btn:
        with st.spinner("Sedang memproses prediksi..."):
            input_vec = vectorizer.transform([st.session_state.input_text_saved])

            pred_sentimen_cat = catboost_sentimen.predict(input_vec.toarray())[0]
            pred_sentimen_gbc = gbc_sentimen.predict(input_vec)[0]

            if y_aspek is not None:
                pred_aspek_cat = catboost_aspek.predict(input_vec.toarray())[0]
                pred_aspek_gbc = gbc_aspek.predict(input_vec)[0]
            else:
                pred_aspek_cat = pred_aspek_gbc = "Unknown"

            aspects_by_keyword = detect_aspects_rule_based(st.session_state.input_text_saved)
            sentiment_by_keyword = detect_sentiment_rule_based(st.session_state.input_text_saved)

            if selected_model == "CatBoost":
                final_sentimen = sentiment_by_keyword if sentiment_by_keyword != "Netral" else pred_sentimen_cat
                final_aspek = aspects_by_keyword if aspects_by_keyword != ["Unknown"] else [pred_aspek_cat]
            elif selected_model == "GradientBoosting":
                final_sentimen = sentiment_by_keyword if sentiment_by_keyword != "Netral" else pred_sentimen_gbc
                final_aspek = aspects_by_keyword if aspects_by_keyword != ["Unknown"] else [pred_aspek_gbc]
            else:
                model_sentimen_vote = pred_sentimen_cat if pred_sentimen_cat == pred_sentimen_gbc else "Netral"
                final_sentimen = sentiment_by_keyword if sentiment_by_keyword != "Netral" else model_sentimen_vote

                if aspects_by_keyword != ["Unknown"]:
                    final_aspek = aspects_by_keyword
                elif pred_aspek_cat == pred_aspek_gbc:
                    final_aspek = [pred_aspek_cat]
                else:
                    final_aspek = ["Gabungan"]

            st.session_state.data_pred_per_model[selected_model] = pd.concat([
                st.session_state.data_pred_per_model[selected_model],
                pd.DataFrame([{
                    "Ulasan": st.session_state.input_text_saved,
                    "Aspek": ", ".join(final_aspek),
                    "Sentimen": final_sentimen
                }])
            ], ignore_index=True)

# --- Display Results ---
st.subheader("Hasil Prediksi per Model")

if "data_pred_per_model" not in st.session_state:
    st.session_state.data_pred_per_model = {
        "GradientBoosting": pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"]),
        "CatBoost": pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"]),
        "Gabungan (Voting)": pd.DataFrame(columns=["Ulasan", "Aspek", "Sentimen"]),
    }

# for model_name, df_model in st.session_state.data_pred_per_model.items():
#     st.write(f"### {model_name}")
#     if not df_model.empty:
#         st.dataframe(df_model.style.set_properties(**{
#             'white-space': 'pre-wrap',
#             'word-wrap': 'break-word'
#         }), hide_index=True, use_container_width=True)
#     else:
#         st.write("Belum ada prediksi untuk model ini.")

for model_name, df_model in st.session_state.data_pred_per_model.items():
    st.write(f"### {model_name}")
    if not df_model.empty:
        st.dataframe(
            df_model.style.set_table_styles([{
                'selector': 'td.col0',
                'props': [
                    ('max-width', '300px'),
                    ('white-space', 'pre-wrap'),
                    ('word-wrap', 'break-word')
                ]
            }]),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.write("Belum ada prediksi untuk model ini.")

