import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, box
from shapely.affinity import translate
import io
from PIL import Image

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("📐 Footwear Material Yield & Layout Visualizer (ProCost Style)")
st.caption("Simulasi Penataan Pola (Nesting) & Calculations Real Waste Material")

# --- SIDEBAR PARAMETER SHEET ---
st.sidebar.header("⚙️️ Parameter Lembaran Material")
sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
inter_gap = st.sidebar.number_input("Jarak Antar Pola / Interlacing Gap (cm)", value=0.5, step=0.1)

# --- FUNGSI OPENCV UNTUK AMBIL POLYGON POLA ---
def extract_polygons_from_image(uploaded_file, dpi=96):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        return []

    if len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pixels_per_cm = dpi / 2.54
    extracted_polygons = []
    
    img_h, img_w = gray.shape[:2]
    max_area_px = (img_h * img_w) * 0.9

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px > 300 and area_px <= max_area_px:
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            pts = approx.reshape(-1, 2) / pixels_per_cm
            if len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    minx, miny, _, _ = poly.bounds
                    poly_zeroed = translate(poly, xoff=-minx, yoff=-miny)
                    extracted_polygons.append(poly_zeroed)

    return extracted_polygons

# --- ALGORITMA SIMULASI NESTING ---
def run_nesting_simulation(polygons, sheet_w, sheet_l, margin_cm, gap_cm):
    placed_polygons = []
    total_pattern_area = 0.0
    
    curr_x = margin_cm
    curr_y = margin_cm
    row_max_h = 0.0

    for idx, poly in enumerate(polygons):
        poly_w = poly.bounds[2] - poly.bounds[0]
        poly_h = poly.bounds[3] - poly.bounds[1]

        if curr_x + poly_w > (sheet_w - margin_cm):
            curr_x = margin_cm
            curr_y += row_max_h + gap_cm
            row_max_h = 0.0

        if curr_y + poly_h > (sheet_l - margin_cm):
            break

        placed_poly = translate(poly, xoff=curr_x, yoff=curr_y)
        placed_polygons.append((placed_poly, idx))
        
        total_pattern_area += placed_poly.area
        curr_x += poly_w + gap_cm
        if poly_h > row_max_h:
            row_max_h = poly_h

    return placed_polygons, total_pattern_area

# --- UI APP ---
uploaded_file = st.file_uploader("Upload Gambar Pattern Component (Master / Multi Pola)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if st.button("🚀 Run Material Yield & Layout Simulation", type="primary"):
        with st.spinner("Mengekstrak geometri & memproses simulasi nesting..."):
            raw_polygons = extract_polygons_from_image(uploaded_file)
            
            if not raw_polygons:
                st.error("Gagal mendeteksi bentuk pola dari gambar. Pastikan kontur garis pola jelas.")
            else:
                placed, net_area = run_nesting_simulation(
                    raw_polygons, sheet_width, sheet_length, margin, inter_gap
                )
                
                total_sheet_area = sheet_width * sheet_length
                gross_yield = (net_area / total_sheet_area) * 100
                total_waste = 100.0 - gross_yield

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Komponen Terpasang", f"{len(placed)} pcs")
                col2.metric("Total Net Area", f"{net_area:.1f} cm²")
                col3.metric("Material Yield", f"{gross_yield:.2f} %")
                col4.metric("Real Cutting Waste", f"{total_waste:.2f} %")

                st.markdown("---")
                st.subheader("🖼️ ProCost-Style Layout Nesting Result")

                fig, ax = plt.subplots(figsize=(12, 8))
                
                sheet_rect = patches.Rectangle((0, 0), sheet_width, sheet_length, linewidth=2, edgecolor='black', facecolor='#F5F5F5')
                ax.add_patch(sheet_rect)
                
                margin_rect = patches.Rectangle((margin, margin), sheet_width - (2*margin), sheet_length - (2*margin), 
                                                linewidth=1, edgecolor='red', linestyle='--', facecolor='none', label='Margin Limit')
                ax.add_patch(margin_rect)

                colors = plt.cm.Set3(np.linspace(0, 1, max(len(raw_polygons), 1)))

                for poly, orig_idx in placed:
                    x, y = poly.exterior.xy
                    c = colors[orig_idx % len(colors)]
                    ax.fill(x, y, alpha=0.8, fc=c, ec='black', linewidth=1.2)
                    
                    centroid = poly.centroid
                    ax.text(centroid.x, centroid.y, f"P{orig_idx+1}", fontsize=8, ha='center', va='center', weight='bold')

                ax.set_xlim(-5, sheet_width + 5)
                ax.set_ylim(-5, sheet_length + 5)
                ax.set_aspect('equal')
                plt.title(f"Visual Optimization Layout (Sheet: {sheet_width}x{sheet_length} cm) | Yield: {gross_yield:.1f}%", fontsize=12)
                plt.xlabel("Width (cm)")
                plt.ylabel("Length (cm)")
                
                st.pyplot(fig)
