import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# -------------------- PAGE CONFIG & CSS --------------------
def config_page():
    st.set_page_config(page_title="Hairtype Detection", layout="wide")
    st.markdown("""
    <style>
    /* Sidebar background & text */
    section[data-testid="stSidebar"] {
        background-color: #800000 !important;
        color: white !important;
    }

    /* Semua teks & label di sidebar */
    section[data-testid="stSidebar"] * {
        color: white !important;
        font-weight: regular !important;
        opacity: 1 !important;
    }

    /* Opsi radio yang dipilih */
    section[data-testid="stSidebar"] div[data-selected="true"] {
        background-color: #A52A2A !important;
        border-radius: 5px;
    }

    /* Hover efek */
    section[data-testid="stSidebar"] div[role="radiogroup"] > div:hover {
        background-color: #993333 !important;
        cursor: pointer;
    }

    /* Teks "Navigasi" */
    .sidebar-title {
        color: white !important;
        font-size: 22px;
        font-weight: bold;
        text-align: left;
        margin-top: 70px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)


# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_model():
    return YOLO("hair_yolobest.pt")

# -------------------- HAIRCARE RECOMMENDATION --------------------
def get_haircare_info(label):
    info = {
        "straight": {
            "deskripsi": "Tipe rambut lurus adalah tipe rambut yang jatuh lembut dari akar hingga ujung dengan kilau alami karena minyak mudah tersebar. Namun, jenis ini mudah lepek, kurang bervolume, dan sulit mempertahankan gaya bergelombang atau keriting..",
            "perawatan": "Gunakan sampo ringan & hindari produk berat.",
        },
        "wavy": {
            "deskripsi": "Rambut bergelombang memiliki bentuk “S” yang muncul di bagian tengah hingga ujung rambut, dan cenderung memiliki volume alami lebih banyak dari rambut lurus. Tantangannya adalah mudah kusut, rentan mengembang (frizzy), serta gelombangnya bisa tidak konsisten.",
            "perawatan": "Gunakan sampo bebas sulfat & kondisioner lembap.",
        },
        "curly": {
            "deskripsi": "Rambut ikal memiliki pola keriting yang terlihat jelas, terutama saat kering. Saat basah, rambut bisa tampak lebih lurus namun akan kembali ikal saat mengering. Jenis rambut ini cenderung mudah mengembang, kering, patah, dan susah diatur.",
            "perawatan": "Gunakan 'squish to condish' & handuk microfiber.",
        },
        "coily": {
            "deskripsi": " Rambut ini memiliki pola keriting sangat rapat, berbentuk spiral kecil atau zigzag, dengan tekstur mulai dari kasar hingga sangat kasar. Meskipun terlihat tebal, rambut ini sangat rapuh, mudah kusut, dan rentan rusak jika terlalu sering disisir atau terkena panas berlebih.",
            "perawatan": "Lakukan deep conditioning mingguan, metode LOC.",
        }
    }
    return info.get(label.lower(), {
        "deskripsi": "Informasi tidak tersedia.",
        "perawatan": "Informasi tidak tersedia.",
        "styling": "Informasi tidak tersedia."
    })


# -------------------- UI COMPONENTS --------------------
def render_sidebar():
    st.sidebar.markdown('<div class="sidebar-title">Navigasi</div>', unsafe_allow_html=True)
    return st.sidebar.radio("Pilih Halaman", ["Beranda", "Deteksi", "Informasi Tipe Rambut"])

def render_footer():
    st.markdown("""
    <style>
    footer {
        display: none;
    }
    .reportview-container .main footer, .stApp {
        padding-bottom: 0px;
        margin-bottom: 0px;
    }
    .block-container {
        padding-bottom: 0px !important;
    }
    </style>
    
    <hr style="border: none; border-top: 1px solid #ccc; margin-top: 50px;"/>
    <div style="text-align: center; padding:10px 0 5px 0; color: red;">
        <p style="margin: 0; font-size: 16px;">&copy; 2025 <strong>Hairtype Detection</strong> — Geldrin Reawaruw</p>
        <p style="margin: 5px 0; font-size: 15px;">
            <a href="https://github.com/" target="_blank" style="color: red; text-decoration: none; margin: 0 10px;">GitHub</a> |
            <a href="https://www.instagram.com/gldrin_reawaruw/" target="_blank" style="color: red; text-decoration: none; margin: 0 10px;">Instagram</a> |
            <a href="https://www.linkedin.com/in/geldrin-reawaruw-545230222/" target="_blank" style="color: red; text-decoration: none; margin: 0 10px;">LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------- PAGE: BERANDA --------------------
def render_beranda():
    st.markdown("<h1 style='text-align:center;'>APLIKASI DETEKSI TIPE RAMBUT MANUSIA</h1>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    col_text, col_img = st.columns([2, 0.9])
    with col_text:
        st.markdown("""
            <div style='font-size:20px; line-height:1.6; text-align:justify;'>
            Aplikasi ini adalah alat berbasis kecerdasan buatan (AI) yang membantu kamu mengetahui jenis rambutmu—lurus, bergelombang, keriting, atau sangat keriting—hanya dengan mengunggah foto. Sistem akan menganalisis bentuk dan tekstur rambutmu secara otomatis, lalu menampilkan hasilnya dalam hitungan detik.  
            <br><br>
            Mengetahui jenis rambut sangat penting karena setiap tipe rambut membutuhkan perawatan yang berbeda. Dengan aplikasi ini, kamu tidak hanya bisa mengenali jenis rambutmu, tapi juga mendapatkan rekomendasi produk dan cara perawatan yang paling sesuai.
            </div>
        """, unsafe_allow_html=True)

    with col_img:
        st.image("img/samping.jpg", caption="Contoh deteksi rambut", use_container_width=True)

    # Bagian Fitur
    st.markdown("<h3 style='text-align:center; margin-top:40px;'>FITUR</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style='background-color:#fff; padding:12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.06); text-align:center;'>
            <img src="https://cdn-icons-png.flaticon.com/512/159/159604.png" width="40" style='margin-bottom:12px;'/>
            <h5 style='color:#800000; margin-bottom:6px;'>Upload Gambar</h5>
            <p style='font-size:14px; text-align:justify;'>Kamu bisa mengunggah gambar rambut dari perangkatmu, lalu sistem akan secara otomatis menganalisis bentuk dan tekstur rambut untuk menentukan jenis rambut yang dimiliki.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color:#fff; padding:12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.06); text-align:center;'>
            <img src="https://cdn-icons-png.flaticon.com/512/747/747376.png" width="40" style='margin-bottom:12px;'/>
            <h5 style='color:#800000; margin-bottom:6px;'>Webcam Real-Time</h5>
            <p style='font-size:14px; text-align:justify;'>Kamu dapat langsung menggunakan kamera webcam untuk mendeteksi tipe rambut secara real-time tanpa perlu unggah file terlebih dahulu.</p>
        </div>
        """, unsafe_allow_html=True)


# -------------------- PAGE: DETEKSI --------------------
def render_deteksi(model):
    st.markdown("<h1 style='text-align:center;'>DETEKSI TIPE RAMBUT MANUSIA</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Upload Gambar", "Kamera"])

    with tab1:
        conf = st.slider("Confidence (%)", 10, 100, 50)
        uploaded = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            img_np = np.array(image)
            results = model.predict(img_np, conf=conf/100)
            result_img = results[0].plot()

            # Tampilkan Gambar Asli & Hasil Deteksi
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Gambar Asli", use_container_width=True)
            with col2:
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

            boxes = results[0].boxes
            if boxes and boxes.cls.numel() > 0:
                class_ids = boxes.cls.cpu().numpy().astype(int)
                labels = list(dict.fromkeys([results[0].names[c] for c in class_ids]))

                # Container untuk hasil deteksi
                st.markdown("""
                    <div style='border: 3px solid #3399ff; border-radius: 15px; padding: 20px; margin-top: 20px; background-color: #f7f9fd;'>
                        <h3 style='text-align:center; color:#003366;'>Tipe Rambut Terdeteksi</h3>
                """, unsafe_allow_html=True)

                # Tampilkan dalam grup kolom 2 per baris
                for i in range(0, len(labels), 2):
                    cols = st.columns([1,1])
                    for j in range(2):
                        if i + j < len(labels):
                            label = labels[i + j]
                            info = get_haircare_info(label)
                            video_urls = {
                                "straight": "7287618275112996102",
                                "wavy": "7497634254172458247",
                                "curly": "7425542102844476678",
                                "coily": "7258012818312809774"
                            }
                            video_embed = f'<iframe src="https://www.tiktok.com/embed/{video_urls.get(label.lower(), "")}" width="100%" height="530" frameborder="0" allowfullscreen></iframe>'

                            with cols[j]:
                                st.markdown(f"""
                                    <div style='background-color:#fff; border-radius:10px; padding:15px; box-shadow: 2px 2px 10px #ccc;'>
                                        <h4 style='color:#800000;'>Tipe: {label.capitalize()}</h4>
                                        <p style='margin-bottom:10px; font-size:18px; text-align: justify;'>{info['deskripsi']}</p>
                                        <p style='margin-bottom:10px; font-size:18px; text-align: justify;'><strong>Tips Perawatan:</strong> {info['perawatan']}</p>
                                        {video_embed}
                                    </div>
                                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)  # Tutup container

            else:
                st.warning("Tidak ada rambut terdeteksi.")


    with tab2:
        st.markdown("### Deteksi Kamera Real-Time")
        conf = st.slider("Confidence Kamera (%)", 10, 100, 50, key="conf_cam")

        # Warna-warna yang akan dipakai bergantian
        colors = [
            (0, 255, 0),     # Hijau
            (0, 0, 255),     # Merah
            (255, 0, 0),     # Biru
            (255, 255, 0),   # Kuning
            (255, 0, 255),   # Ungu
            (0, 255, 255),   # Cyan
            (128, 128, 128), # Abu-abu
            (255, 128, 0),   # Orange
        ]


        class HairDetectionProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = model

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")  # dari webcam
                results = self.model(img, conf=conf / 100)[0]

                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_score = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls]

                    # Ambil warna unik berdasarkan indeks i
                    color = colors[i % len(colors)]

                    # Gambar bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{label} {conf_score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                return av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")

        webrtc_streamer(
            key="hairtype-realtime",
            video_processor_factory=HairDetectionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

# -------------------- PAGE: INFORMASI --------------------
def render_info():
    st.markdown("<h1 style='text-align:center;'>INFORMASI TIPE RAMBUT</h1>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:justify; font-size:22px; line-height:1.6;'>
    Rambut manusia memiliki berbagai tipe yang unik dan dipengaruhi oleh faktor genetik, etnis, serta lingkungan.
    Memahami tipe rambut sangat penting untuk menentukan perawatan yang tepat serta untuk pengembangan produk kecantikan atau medis yang sesuai.
    </div><br>
    """, unsafe_allow_html=True)

    def hair_type_box(title, style_name, image_path, index, description):
        col1, col2 = st.columns([1, 1]) if index % 2 == 0 else st.columns([1, 1])

        if index % 2 == 0:
            # Gambar kiri, teks kanan
            with col1:
                st.image(image_path, width=500)
            with col2:
                st.markdown(f"""
                    <div style='background-color:#800000; padding:25px; border-radius:15px; 
                                box-shadow: 2px 2px 6px #444; color:white; margin-bottom:30px; text-align:justify;'>
                        <h4 style='color:white;'>{title} <i style='color:white;'>({style_name})</i></h4>
                        <p style='font-size:18px; line-height:1;'>{description}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            # Teks kiri, gambar kanan
            with col1:
                st.markdown(f"""
                    <div style='background-color:#800000; padding:25px; border-radius:15px; 
                                box-shadow: 2px 2px 6px #444; color:white; margin-bottom:30px; text-align:justify;'>
                        <h4 style='color:white;'>{title} <i style='color:white;'>({style_name})</i></h4>
                        <p style='font-size:18px; line-height:1;'>{description}</p>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.image(image_path, width=500)

    hair_type_box(
        "Tipe Rambut Lurus",
        "Straight",
        "img/straight1.png",
        0,
        """
        Rambut lurus memiliki helai yang jatuh lembut dari akar hingga ujung, dengan kilau alami karena minyak kulit kepala mudah menyebar. 
        Namun, jenis rambut ini cenderung mudah lepek, kurang bervolume, dan sulit mempertahankan gaya rambut bergelombang atau keriting.

        <strong>Kekurangan:
        - Bisa tampak lepek dan kurang bervolume.
        - Rentan terhadap polusi dan cepat terlihat kusam jika tidak dirawat dengan baik.

        <strong>Perawatan:</strong><br>
        - Gunakan sampo yang ringan dan tidak membuat rambut lepek.
        - Gunakan <em>Dove 1 Minute Super Conditioner Hair Fall Rescue</em> untuk membantu mengurangi rambut rontok.
        - Lakukan perawatan mingguan dengan <em>Dove Creambath Hair Growth Ritual</em> untuk menjaga kekuatan akar rambut.

        <strong>Styling:</strong>
        - Gunakan dry shampoo di akar untuk menambah volume.
        - Alat catok bergelombang, rol panas, atau sea salt spray bisa membantu menciptakan tekstur.
        - Gunakan mousse ringan untuk memberikan efek bervolume yang tahan lama.
        """
    )


    hair_type_box(
        "Tipe Rambut Bergelombang",
        "Wavy",
        "img/wavy1.png",
        1,
        """
        Rambut bergelombang memiliki bentuk “S” yang muncul di bagian tengah hingga ujung rambut, dan cenderung memiliki volume alami lebih banyak 
        dari rambut lurus. Tantangannya adalah mudah kusut, rentan mengembang (frizzy), serta gelombangnya bisa tidak konsisten.

        <strong>Perawatan:</strong>
        - Gunakan produk dengan formula pelembap ringan.
        - Setelah keramas, gunakan <em>Dove 1 Minute Super Conditioner Intensive Damage Treatment</em> untuk menghaluskan dan melembapkan rambut.
        - Lakukan creambath seminggu sekali dengan <em>Dove Creambath Hair Growth Ritual</em> untuk nutrisi dan mengunci kelembapan.

        <strong>Styling:</strong>
        - Gunakan metode scrunching atau plopping saat rambut setengah kering. 
        - Keringkan dengan diffuser agar gelombang tetap terbentuk alami. 
        - Tambahkan sea salt spray atau mousse ringan untuk efek bergelombang yang tahan lama.
        """
    )


    hair_type_box(
        "Tipe Rambut Keriting",
        "Curly",
        "img/curly1.png",
        2,
        """
        Rambut ikal memiliki pola keriting yang terlihat jelas, terutama saat kering. Saat basah, 
        rambut bisa tampak lebih lurus namun akan kembali ikal saat mengering. 
        Jenis rambut ini cenderung mudah mengembang, kering, patah, dan susah diatur.

        <strong>Perawatan:</strong>
        - Gunakan sampo yang mengandung argan oil dan vitamin E.
        - Gunakan kondisioner secara rutin untuk menjaga kelembapan lekukan rambut.
        - Aplikasikan kondisioner tanpa bilas setelah keramas.
        - Hindari produk dengan silikon dan asam sulfat.
        - Hindari menyisir dan menguncir rambut terlalu sering.

        <strong>Styling:</strong>
        - Terapkan teknik rake and shake atau finger coiling dengan leave-in conditioner dan curl cream saat rambut setengah basah. 
        - Gunakan diffuser pada suhu rendah untuk mempertahankan bentuk ikal. 
        - Styling gel bisa membantu mempertahankan definisi lebih lama.
        """
    )

    hair_type_box(
        "Tipe Rambut Sangat Keriting",
        "Coily",
        "img/coily1.png",
        3,
        """
        Rambut ini memiliki pola keriting sangat rapat, berbentuk spiral kecil atau zigzag, dengan tekstur mulai dari kasar hingga sangat kasar. 
        Meskipun terlihat tebal, rambut ini sangat rapuh, mudah kusut, dan rentan rusak jika terlalu sering disisir atau terkena panas berlebih.
        
        <strong>Kekurangan:</strong>
        - Rentan terhadap kerusakan akibat panas dan zat kimia.
        - Mudah kusut dan patah bila tidak dirawat dengan hati-hati.

        <strong>Perawatan:</strong>
        - Gunakan kondisioner tanpa bilas dan masker rambut <em>deep conditioning</em>.
        - Gunakan sampo dan kondisioner yang memperbaiki kerusakan.
        - Hindari menyisir terlalu sering dan gunakan produk yang menutrisi dari akar hingga ujung rambut.

        <strong>Styling:</strong>
        - Terapkan gaya pelindung seperti twists, bantu knots, atau box braids untuk menjaga kelembapan dan mengurangi kerusakan. 
        - Teknik twist out atau braid out juga cocok untuk tampilan alami. 
        - Gunakan jari atau sisir bergigi jarang saat menata rambut agar tekstur tidak rusak.
        """
    )

# -------------------- MAIN --------------------
def main():
    config_page()
    menu = render_sidebar()
    model = load_model()

    if menu == "Beranda":
        render_beranda()
    elif menu == "Deteksi":
        render_deteksi(model)
    elif menu == "Informasi Tipe Rambut":
        render_info()

    render_footer()

if __name__ == "__main__":
    main()