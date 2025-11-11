import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Konfigurasi Tampilan
# ==========================
st.set_page_config(
    page_title="UTS Pemrograman Big Data",
    page_icon="🧠",
    layout="centered"
)

# Warna latar belakang biru (CSS)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #b3d9ff; /* biru muda */
        color: #000000; /* teks hitam */
    }
    h1, h2, h3 {
        color: #003366; /* biru tua untuk judul */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Sidebar Menu
# ==========================
menu = st.sidebar.selectbox(
    "Navigasi:",
    ["🏠 Halaman Utama", "🔍 Deteksi Objek (YOLO)", "🖼️ Klasifikasi Gambar", "👩‍💻 Penyusun"]
)

# ==========================
# Halaman Utama
# ==========================
if menu == "🏠 Halaman Utama":
    st.title("Ujian Tengah Semester Pemrograman Big Data")
    st.markdown(
        """
        ### Selamat datang di aplikasi UTS Pemrograman Big Data!
        Aplikasi ini memiliki dua fungsi utama:
        1. **Deteksi Objek (YOLO)** — mendeteksi objek dalam gambar menggunakan model YOLOv8.  
        2. **Klasifikasi Gambar** — mengklasifikasikan gambar berdasarkan model CNN (TensorFlow/Keras).  

        Pilih mode di **sidebar** untuk memulai.
        """
    )

# ==========================
# Halaman Deteksi Objek (YOLO)
# ==========================
elif menu == "🔍 Deteksi Objek (YOLO)":
    st.title("Deteksi Objek Menggunakan YOLO")

    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

# ==========================
# Halaman Klasifikasi Gambar
# ==========================
elif menu == "🖼️ Klasifikasi Gambar":
    st.title("Klasifikasi Gambar Menggunakan CNN")

    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)

        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", np.max(prediction))

# ==========================
# Halaman Penyusun
# ==========================
elif menu == "👩‍💻 Penyusun":
    st.title("👩‍💻 Halaman Penyusun")
    st.markdown(
        """
        ### Disusun oleh:
        **Nama:** Sufia Humaira  
        **NPM:** 2108108010088  
        **Mata Kuliah:** Pemrograman Big Data  
        **Universitas:** Universitas Syiah Kuala  

        ---
        Terima kasih telah menggunakan aplikasi ini 💙  
        """
    )
