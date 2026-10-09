import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide"
)

st.title("⚡ Footwear Material Nesting & Drawing Studio")
st.markdown("---")

# ============================================================
# TAHAP 1: SETUP PARAMETER UTAMA DI HALAMAN UTAMA (COMPACT GRID)
# ============================================================

st.subheader("📋 Material Setup")
col_p1, col_p1_unit, col_p2, col_p3, col_p4 = st.columns(
    [1.5, 1, 1.2, 1.2, 1.2]
)

with col_p1:
  material_width_input = st.number_input(
      "Material Width", value=44.0, step=1.0, format="%.2f"
  )
with col_p1_unit:
  width_unit = st.selectbox("Width Unit", ["inch", "cm", "m"], index=0)

# Konversi lebar material ke cm berdasarkan pilihan unit
if width_unit == "inch":
  sheet_width = material_width_input * 2.54
elif width_unit == "m":
  sheet_width = material_width_input * 100.0
else:
  sheet_width = material_width_input

with col_p2:
  sheet_length = st.number_input("Material Length (cm)", value=100.0, step=5.0)
with col_p3:
  margin = st.number_input("Margin (cm)", value=1.0, step=0.5)
with col_p4:
  target_pairs = st.number_input("Target Pairs", value=50, min_value=1, step=1)
  target_pieces = target_pairs * 2

st.markdown("---")
st.subheader("📥 Input Sumber Pola Komponen")

# Pilihan metode input: Upload File vs Kanvas Gambar Interaktif
input_method = st.radio(
    "Pilih Cara Input Pola:",
    (
        "Upload Gambar Pola (File)",
        "Gambar di Kanvas & Download (HTML Canvas Studio)",
    ),
    horizontal=True,
)

file_bytes = None

if input_method == "Upload Gambar Pola (File)":
  uploaded_file = st.file_uploader(
      "Upload Pattern Component Master Image", type=["png", "jpg", "jpeg"]
  )
  if uploaded_file is not None:
    file_bytes = uploaded_file.read()

else:
  st.markdown(
      "1. Gambar bentuk komponen sepatu pada kanvas di bawah dengan garis"
      " hitam yang tertutup rapat.<br>2. Klik tombol **'💾 Download Gambar"
      " Pola'**.<br>3. Unggah file hasil download tersebut melalui menu"
      " **Upload File** di atas (atau ubah pilihan metode ke Upload).",
      unsafe_allow_html=True,
  )

  # Kanvas HTML5 interaktif dengan tombol download instan
  canvas_html = """
    <div style="background: #fdfdfd; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; display: inline-block;">
        <div style="margin-bottom: 10px; display: flex; gap: 10px; align-items: center;">
            <button id="clearBtn" style="padding: 8px 14px; background-color: #ff4b4b; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">🧹 Clear Canvas</button>
            <button id="downloadBtn" style="padding: 8px 14px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">💾 Download Gambar Pola</button>
            <span id="statusTxt" style="font-size: 13px; color: #555; margin-left: 5px;">(Gunakan garis hitam tebal & tertutup)</span>
        </div>
        <canvas id="paintCanvas" width="700" height="400" style="border:2px solid #ccc; background-color:#ffffff; cursor:crosshair; border-radius: 6px; display: block;"></canvas>
    </div>

    <script>
        const canvas = document.getElementById('paintCanvas');
        const ctx = canvas.getContext('2d');
        let painting = false;

        // Set background putih awal agar tidak transparan
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        function startPosition(e) {
            painting = true;
            draw(e);
        }

        function finishedPosition() {
            painting = false;
            ctx.beginPath();
        }

        function draw(e) {
            if (!painting) return;
            ctx.lineWidth = 4;
            ctx.lineCap = 'round';
            ctx.strokeStyle = '#000000';

            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        // Support Touch untuk HP/Tablet
        function startTouch(e) {
            painting = true;
            drawTouch(e);
            e.preventDefault();
        }

        function drawTouch(e) {
            if (!painting) return;
            ctx.lineWidth = 4;
            ctx.lineCap = 'round';
            ctx.strokeStyle = '#000000';

            const rect = canvas.getBoundingClientRect();
            const touch = e.touches[0];
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;

            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
            e.preventDefault();
        }

        canvas.addEventListener('mousedown', startPosition);
        canvas.addEventListener('mouseup', finishedPosition);
        canvas.addEventListener('mousemove', draw);

        canvas.addEventListener('touchstart', startTouch);
        canvas.addEventListener('touchend', finishedPosition);
        canvas.addEventListener('touchmove', drawTouch);

        document.getElementById('clearBtn').addEventListener('click', function() {
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('statusTxt').innerText = "Canvas dibersihkan.";
        });

        document.getElementById('downloadBtn').addEventListener('click', function() {
            const dataURL = canvas.toDataURL('image/png');
            const link = document.createElement('a');
            link.download = 'pola_komponen_sepatu.png';
            link.href = dataURL;
            link.click();
            document.getElementById('statusTxt').innerText = "✅ Gambar berhasil di-download! Silakan upload file tersebut di atas.";
        });
    </script>
    """

  components.html(canvas_html, height=480)

  # Tambahan widget file uploader khusus untuk hasil gambar kanvas agar langsung diproses
  st.markdown("##### 📤 Masukkan Hasil Gambar Kanvas Anda di Sini:")
  canvas_uploaded_file = st.file_uploader(
      "Upload file 'pola_komponen_sepatu.png' dari hasil download kanvas",
      type=["png", "jpg", "jpeg"],
      key="canvas_uploader",
  )
  if canvas_uploaded_file is not None:
    file_bytes = canvas_uploaded_file.read()

# ============================================================
# EXTRACT POLYGONS FROM IMAGE / CANVAS
# ============================================================


@st.cache_data
def extract_polygons_from_bytes(file_bytes, dpi=96):
  try:
    img = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED
    )
    if img is None:
      return []

    if len(img.shape) == 3 and img.shape[2] == 4:
      alpha = img[:, :, 3]
      rgb = img[:, :, :3]
      background = np.ones_like(rgb, dtype=np.uint8) * 255
      alpha_factor = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
      gray = cv2.cvtColor(
          (rgb * alpha_factor + background * (1 - alpha_factor)).astype(
              np.uint8
          ),
          cv2.COLOR_BGR2GRAY,
      )
    elif len(img.shape) == 3:
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
      gray = img

    blurred = cv2.GaussianBlur(gray,
