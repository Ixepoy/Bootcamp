import streamlit as st
from deepface import DeepFace
import pandas as pd
from PIL import Image
import tempfile
import datetime
import os
import urllib.parse
import base64
from PIL.ExifTags import TAGS
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import cv2
import av
import requests  

# =====================
# FUNGSI DETEKSI & UTILITY
# =====================
def cek_wajah(path):
    try:
        faces = DeepFace.extract_faces(
            img_path=path,
            enforce_detection=False 
        )
        return len(faces) > 0
    except:
        return False

@st.cache_resource
def load_mask_model():
    try:
        model = tf.keras.models.load_model('model_wajah.keras')
        return model
    except Exception as e:
        return None

mask_model = load_mask_model()

def generate_pdf(age, gender, emotion, overview):
    filename = "face_report.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("OSINT Face Intelligence Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    data = [
        ["Parameter", "Hasil"],
        ["Age", str(age)],
        ["Gender", gender],
        ["Emotion", emotion]
    ]

    table = Table(data, colWidths=[150, 250])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige)
        ])
    )

    elements.append(table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("AI Overview", styles["Heading2"]))
    elements.append(Paragraph(overview, styles["BodyText"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now()}", styles["Italic"]))

    doc.build(elements)
    return filename

def get_exif_data(image):
    exif_data = {}
    try:
        exif = image.getexif()
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            exif_data[tag] = value
    except:
        pass
    return exif_data


# ==========================================
# KONFIGURASI & CALLBACK LIVE CAMERA (WebRTC)
# ==========================================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Target Terdeteksi", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# =====================
# CONFIG DASHBOARD
# =====================
st.set_page_config(
    page_title="OSINT Face Intelligence Dashboard",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ OSINT Face Intelligence Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Face Analysis",
    "Face Matching",
    "Live Camera",
    "Riwayat Analisis"
])

# ==================================================
# FACE ANALYSIS (TAB 1)
# ==================================================
with tab1:
    st.header("Analisis Wajah")

    sumber = st.radio("Pilih Sumber Gambar", ["Upload File", "Kamera"])
    if sumber == "Upload File":
        uploaded = st.file_uploader("Upload Foto Wajah", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.camera_input("Ambil Foto")

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, width=300)

        with st.expander("📷 Lihat Metadata Foto"):
            metadata = get_exif_data(image)
            if metadata:
                df_meta = pd.DataFrame(list(metadata.items()), columns=['Tag', 'Value'])
                st.dataframe(df_meta, use_container_width=True, hide_index=True)
            else:
                st.info("Metadata EXIF tidak ditemukan pada gambar ini.")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.getbuffer())
            img_path = tmp.name

        if st.button("Analisis Wajah"):
            if not cek_wajah(img_path):
                st.error("❌ Gambar bukan wajah manusia atau wajah tidak terdeteksi.")
            else:
                try:
                    result = DeepFace.analyze(
                        img_path=img_path,
                        actions=["age", "gender", "emotion"],
                        enforce_detection=False
                    )

                    data = result[0]
                    st.success("✅ Analisis Selesai")
                    
                    status_masker = "Tidak Diketahui"
                    if mask_model is not None:
                        try:
                            img_test = load_img(img_path, target_size=(128, 128))
                            img_array = img_to_array(img_test)
                            img_array = np.expand_dims(img_array, axis=0)
                            img_array = img_array / 255.0

                            prediction = mask_model.predict(img_array, verbose=0)
                            if prediction[0][0] > 0.5:
                                status_masker = "Tanpa Masker ❌"
                            else:
                                status_masker = "Menggunakan Masker 😷"
                        except Exception as e:
                            status_masker = "Gagal Deteksi"
                            st.warning(f"AI Masker melewati gambar ini. Info: {e}")
                    else:
                        status_masker = "Tidak Diketahui"

                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Umur", data["age"])
                    with col2: st.metric("Gender", data["dominant_gender"])
                    with col3: st.metric("Emosi", data["dominant_emotion"])
                    with col4: st.metric("Status Masker", status_masker)

                    st.subheader("🤖 AI Overview")
                    overview = f"""
                    Berdasarkan analisis AI:
                    • Umur diperkirakan sekitar {data['age']} tahun.
                    • Gender dominan terdeteksi sebagai {data['dominant_gender']}.
                    • Emosi dominan yang terlihat adalah {data['dominant_emotion']}.
                    • Status Masker: {status_masker}.
                    
                    Catatan: Hasil ini merupakan estimasi AI dan tidak dapat digunakan
                    untuk mengidentifikasi identitas seseorang secara pasti.
                    """
                    st.info(overview)
                   
                    pdf_file = generate_pdf(data["age"], data["dominant_gender"], data["dominant_emotion"], overview)
                    with open(pdf_file, "rb") as f:
                        st.download_button("📄 Download PDF Report", f, "report.pdf", "application/pdf")
                    
                    log = pd.DataFrame([{
                        "timestamp": datetime.datetime.now(),
                        "age": data["age"],
                        "gender": data["dominant_gender"],
                        "emotion": data["dominant_emotion"]
                    }])

                    if os.path.exists("logs.csv"):
                        old = pd.read_csv("logs.csv")
                        log = pd.concat([old, log], ignore_index=True)

                    log.to_csv("logs.csv", index=False)

                except Exception as e:
                    st.error(f"Terjadi error: {e}")

# ==================================================
# FACE MATCHING (TAB 2)
# ==================================================
with tab2:
    st.header("Face Matching")
    st.subheader("Target")
    
    target_source = st.radio("Sumber Target", ["Upload", "Kamera"], key="target_source")
    
    if target_source == "Upload":
        img1 = st.file_uploader("Upload Target", type=["jpg", "jpeg", "png"], key="img1")
    else:
        img1 = st.camera_input("Foto Target", key="cam1")

    st.subheader("Pembanding")
    compare_source = st.radio("Sumber Pembanding", ["Upload", "Kamera"], key="compare_source")

    if compare_source == "Upload":
        img2 = st.file_uploader("Upload Pembanding", type=["jpg", "jpeg", "png"], key="img2")
    else:
        img2 = st.camera_input("Foto Pembanding", key="cam2")

    if img1 and img2:
        col1, col2 = st.columns(2)
        with col1: st.image(img1, caption="Target")
        with col2: st.image(img2, caption="Pembanding")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t1:
            t1.write(img1.getbuffer())
            path1 = t1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t2:
            t2.write(img2.getbuffer())
            path2 = t2.name

        if st.button("Bandingkan Wajah"):
            if not cek_wajah(path1):
                st.error("❌ Target bukan wajah manusia.")
            elif not cek_wajah(path2):
                st.error("❌ Gambar pembanding bukan wajah manusia.")
            else:
                try:
                    result = DeepFace.verify(img1_path=path1, img2_path=path2, enforce_detection=False)
                    distance = result["distance"]

                    if result["verified"]:
                        st.success("✅ Wajah Cocok")
                    else:
                        st.error("❌ Wajah Tidak Cocok")

                    similarity = (1 - distance) * 100
                    if similarity < 0: similarity = 0

                    st.metric("Similarity Score", f"{similarity:.2f}%")
                    st.subheader("🤖 AI Overview")

                    overview = f"""
                    Hasil analisis menunjukkan:
                    • Similarity Score: {similarity:.2f}%
                    • Distance: {distance:.4f}
                    Semakin kecil nilai distance, semakin besar kemungkinan kedua
                    gambar berasal dari orang yang sama.
                    """
                    st.info(overview)

                except Exception as e:
                    st.error(f"Terjadi error: {e}")

# ==================================================
# LIVE CAMERA (TAB 3)
# ==================================================
with tab3:
    st.header("📷 Live Camera")
    st.info("Klik tombol 'Start' di bawah untuk mengaktifkan kamera.")

    webrtc_streamer(
        key="camera-live-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ==================================================
# RIWAYAT ANALISIS (TAB 4)
# ==================================================
with tab4:
    st.header("📜 Riwayat Analisis")
    if os.path.exists("logs.csv"):
        df = pd.read_csv("logs.csv")
        page_size = 5
        total_rows = len(df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)

        page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        st.dataframe(df.iloc[start_idx:end_idx])
        st.caption(f"Menampilkan data {start_idx+1} - {min(end_idx, total_rows)} dari {total_rows}")
        st.download_button("⬇ Download Log CSV", df.to_csv(index=False), "logs.csv")
    else:
        st.info("Belum ada riwayat analisis.")

# ==================================================
# SOCIAL MEDIA INTELLIGENCE (SOCMINT)
# ==================================================
st.divider()
st.header("🌐 Social Media Intelligence (SOCMINT)")
st.write("Cari jejak digital target menggunakan gambar wajah atau informasi teks.")

socmint_tab1, socmint_tab2 = st.tabs(["🖼️ Reverse Image Search", "🔎 Text & Dorking Search"])

# -----------------------------------------
# TAB SOCMINT 1: REVERSE IMAGE SEARCH
# -----------------------------------------
with socmint_tab1:
    st.info("Gunakan API OpenWebNinja atau mesin pencari publik untuk menemukan jejak digital wajah target.")
    
    socmint_img = st.file_uploader("Upload Foto Target", type=["jpg", "jpeg", "png"], key="socmint_img")
    
    if socmint_img:
        st.image(socmint_img, caption="Foto Target", width=300)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t_socmint:
            t_socmint.write(socmint_img.getbuffer())
            path_socmint = t_socmint.name
            
        st.success("Foto siap digunakan.")

        # =====================================================
        # FITUR UTAMA: API OPENWEBNINJA DENGAN MULTI-UPLOAD FALLBACK
        # =====================================================
        st.divider()
        st.subheader("🤖 Otomatis: Pencarian via API OpenWebNinja")
        
        if st.button("🔍 Jalankan Reverse Image API"):
            image_url = None
            
            # --- TAHAP 1: PROSES UPLOAD MULTI-SERVER ---
            with st.spinner("1️⃣ Mengunggah foto target ke server cadangan sementara..."):
                
                # Kandidat 1: Tmpfiles.org 
                try:
                    with open(path_socmint, "rb") as f:
                        resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f}, timeout=10)
                        if resp.status_code == 200:
                            res_json = resp.json()
                            raw_url = res_json.get("data", {}).get("url", "")
                            if raw_url:
                                image_url = raw_url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/")
                except Exception as e:
                    pass 
                
                # Kandidat 2: Catbox.moe 
                if not image_url:
                    try:
                        with open(path_socmint, "rb") as f:
                            resp = requests.post("https://catbox.moe/user/api.php", data={"reqtype": "fileupload"}, files={"fileToUpload": f}, timeout=10)
                            if resp.status_code == 200 and "https://" in resp.text:
                                image_url = resp.text.strip()
                    except Exception as e:
                        pass

                # Kandidat 3: File.io 
                if not image_url:
                    try:
                        with open(path_socmint, "rb") as f:
                            resp = requests.post("https://file.io", files={"file": f}, timeout=10)
                            if resp.status_code == 200:
                                image_url = resp.json().get("link")
                    except Exception as e:
                        pass

            # --- TAHAP 2: KIRIM URL KE OPENWEBNINJA ---
            if image_url:
                st.info(f"🔗 Link publik foto diperoleh: {image_url}")
                with st.spinner("2️⃣ Mengirim foto publik ke mesin OpenWebNinja..."):
                    try:
                        url = "https://api.openwebninja.com/reverse-image-search/reverse-image-search"
                        querystring = {"url": image_url}
                        headers = {
                          "X-API-Key": "ak_av7ckg8ff8r1o0xmj7xulsl6w2s5a7nruyjmhox40xx1lti"
                        }
                        
                        response = requests.get(url, headers=headers, params=querystring, timeout=80)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Mengekstrak hasil dari JSON yang strukturnya dinamis
                            result_data = data.get("data", data)
                            matches = []
                            
                            # Logika pintar untuk menarik data array meskipun nama key-nya kosong ("")
                            if isinstance(result_data, dict):
                                if "visual_matches" in result_data:
                                    matches = result_data["visual_matches"]
                                elif "" in result_data and isinstance(result_data[""], list):
                                    matches = result_data[""]
                                else:
                                    # Fallback: Cari array/list apa pun di dalam data
                                    for key, value in result_data.items():
                                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                                            matches.extend(value)
                            elif isinstance(result_data, list):
                                matches = result_data

                            if matches:
                                st.success(f"✅ Pelacakan Selesai! Ditemukan {len(matches)} jejak digital.")
                                st.subheader("🌐 Website yang memiliki gambar serupa")
                                
                                # Menampilkan dalam bentuk modern list/card
                                for item in matches:
                                    link = item.get("link", item.get("url", "#"))
                                    title = item.get("title", "Tidak ada judul")
                                    thumbnail = item.get("thumbnail", item.get("image", ""))
                                    
                                    # Mengambil nama domain website secara rapi
                                    domain_source = urllib.parse.urlparse(link).netloc
                                    source = item.get("source", item.get("domain", domain_source))
                                    
                                    # Membuat antarmuka bergaya "Card" dengan container bergaris
                                    with st.container(border=True):
                                        col_img, col_info = st.columns([1, 5]) # Tata letak lebar kolom
                                        
                                        with col_img:
                                            if thumbnail:
                                                st.image(thumbnail, use_container_width=True)
                                            else:
                                                st.markdown("<h3>🌐</h3>", unsafe_allow_html=True)
                                                
                                        with col_info:
                                            st.markdown(f"**{title}**")
                                            st.caption(f"Sumber: `{source}`")
                                            # Membuat tombol full-width
                                            st.link_button("Kunjungi Situs Pelacakan ↗️", link, use_container_width=True)
                            else:
                                st.info("Tidak ada kecocokan situs web spesifik di respon API.")
                                
                            # Opsi bagi analis untuk tetap melihat struktur asli log-nya (disembunyikan)
                            with st.expander("Lihat Respons JSON Mentah (Untuk Tim Analis)", expanded=False):
                                st.json(data)
                                
                        else:
                            st.error(f"❌ API Error OpenWebNinja: {response.status_code} - {response.text}")
                    
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses OpenWebNinja: {e}")
            else:
                st.error("❌ Semua server unggah gambar sementara (Tmpfiles, Catbox, File.io) gagal merespon. Silakan periksa koneksi internet Anda atau gunakan pencarian manual di bawah.")

        # =====================================================
        # PENCARIAN MANUAL (BACKUP)
        # =====================================================
        st.divider()
        st.subheader("🔗 Manual: Pencarian Mesin Publik")
        st.write("Gunakan drag-and-drop jika sistem API di atas sedang mencapai batas limit (kuota).")
        
        col_yandex, col_google, col_bing = st.columns(3)
        with col_yandex:
            st.link_button("Cari di Yandex Images ↗️", "https://yandex.com/images/", use_container_width=True)
        with col_google:
            st.link_button("Cari di Google Lens ↗️", "https://images.google.com/", use_container_width=True)
        with col_bing:
            st.link_button("Cari di Bing Visual ↗️", "https://www.bing.com/images/feed", use_container_width=True)
            

# -----------------------------------------
# TAB SOCMINT 2: TEXT & DORKING SEARCH
# -----------------------------------------
with socmint_tab2:
    st.write("Gunakan teknik *Google Dorking* untuk menemukan profil media sosial berdasarkan Nama atau Username.")
    
    nama_target = st.text_input("Masukkan Nama atau Username Target", key="nama_dorking")
    
    if nama_target:
        encoded_name = urllib.parse.quote(nama_target)
        
        dork_links = {
            "Google Umum": f"https://www.google.com/search?q={encoded_name}",
            "LinkedIn Profile": f"https://www.google.com/search?q={encoded_name}+site:linkedin.com/in/",
            "Instagram Profile": f"https://www.google.com/search?q={encoded_name}+site:instagram.com",
            "Facebook Profile": f"https://www.google.com/search?q={encoded_name}+site:facebook.com",
            "X (Twitter) Profile": f"https://www.google.com/search?q={encoded_name}+site:twitter.com OR site:x.com",
            "GitHub Profile": f"https://www.google.com/search?q={encoded_name}+site:github.com"
        }
        
        st.subheader("🔗 Hasil Generasi Link Dorking:")
        for platform, url in dork_links.items():
            st.link_button(f"Cari di {platform}", url)
