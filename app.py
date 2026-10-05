import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate
import io
from PIL import Image

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("📐 Footwear Material Yield & Layout Visualizer (ProCost Style)")
st.caption("Simulasi True Shape Nesting (Interlock) & Complete Material Consumption Metrics")

# --- SIDEBAR PARAMETER SHEET ---
st.sidebar.header("⚙️ Parameter Lembaran Material")
sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
inter_gap = st.sidebar.number_input("Jarak Antar Pola / Interlacing Gap (cm)", value=0.5, step=0.1)
target_pairs = st.sidebar.number_input("Jumlah Pasang Target (Pairs)", value=50, min_value=1, step=1)

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

# --- ALGORITMA NESTING ADVANCED (TRUE INTERLOCK) ---
def run_true_shape_nesting(polygons, sheet_w, sheet_l, margin_cm, gap_cm, total_pairs):
    placed_polygons = []
    total_pattern_area = 0.0
    
    # Buat pool komponen (Pairs x Jumlah Komponen)
    pattern_pool = []
    for pair in range(total_pairs):
        for idx, poly in enumerate(polygons):
            pattern_pool.append((poly, idx))

    curr_x = margin_cm
    curr_y = margin_cm
    row_max_h = 0.0
    placed_in_row = []

    for item_idx, (poly, orig_idx) in enumerate(pattern_pool):
        # Rotasi selang-seling 180 derajat untuk pasangan interlock
        best_poly = poly
        if item_idx % 2 == 1:
            best_poly = rotate(poly, 180, origin='center')
            minx, miny, _, _ = best_poly.bounds
            best_poly = translate(best_poly, xoff=-minx, yoff=-miny)

        poly_w = best_poly.bounds[2] - best_poly.bounds[0]
        poly_h = best_poly.bounds[3] - best_poly.bounds[1]

        # Cari posisi X terapat tanpa bertabrakan dengan pola sebelumnya
        test_x = curr_x
        if placed_in_row:
            # Geser sedikit demi sedikit untuk mengecek interlock geometri
            min_shift = 0.1
            candidate_x = curr_x
            
            # Geser mundur untuk interlock rapat
            for step_x in np.arange(curr_x, curr_x - (poly_w * 0.7), -0.2):
                test_poly = translate(best_poly, xoff=step_x, yoff=curr_y)
                # Cek overlap dengan komponen sebelumnya di baris yang sama
                has_overlap = False
                for prev_p, _ in placed_in_row:
                    if test_poly.buffer(gap_cm / 2).intersects(prev_p):
                        has_overlap = True
                        break
                if not has_overlap:
                    candidate_x = step_x
                else:
                    break
            test_x = max(candidate_x, margin_cm)

        # Cek batas kanan lembaran
        if test_x + poly_w > (sheet_w - margin_cm):
            # Pindah Baris Baru ke Atas
            curr_x = margin_cm
            curr_y += row_max_h + gap_cm
            row_max_h = 0.0
            placed_in_row = []
            test_x = margin_cm

        # Cek batas atas lembaran
        if curr_y + poly_h > (sheet_l - margin_cm):
            break  # Kain penuh

        placed_poly = translate(best_poly, xoff=test_x, yoff=curr_y)
        placed_polygons.append((placed_poly, orig_idx))
        placed_in_row.append((placed_poly, orig_idx))
        
        total_pattern_area += placed_poly.area
        curr_x = test_x + poly_w + gap_cm
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
                placed, net_area = run_true_shape_nesting(
                    raw_polygons, sheet_width, sheet_length, margin, inter_gap, target_pairs
                )
                
                total_sheet_area = sheet_width * sheet_length
                
                # Hitung Area Kain Efektif Terpakai (sampai batas tinggi komponen tertinggi)
                max_used_y = max([p.bounds[3] for p, _ in placed]) if placed else 0.0
                used_sheet_area = sheet_width * max_used_y if max_used_y > 0 else total_sheet_area

                # METRIK KALKULASI YIELD
                component_yield = (net_area / used_sheet_area) * 100 if used_sheet_area > 0 else 0.0
                overall_sheet_yield = (net_area / total_sheet_area) * 100
                total_waste = 100.0 - component_yield
                
                pairs_completed = len(placed) // len(raw_polygons) if raw_polygons else 0
                consumption_per_pair = (used_sheet_area / 10000) / max(pairs_completed, 1) # dalam m² / pair

                # --- DASHBOARD METRIK PROCOST ---
                st.markdown("### 📊 Yield & Material Consumption Summary")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Komponen Terpasang", f"{len(placed)} pcs ({pairs_completed} pairs)")
                m2.metric("Total Net Area", f"{net_area:.1f} cm²")
                m3.metric("Component Yield", f"{component_yield:.2f} %")
                m4.metric("Overall Sheet Yield", f"{overall_sheet_yield:.2f} %")
                m5.metric("Cutting Waste", f"{total_waste:.2f} %")

                st.info(f"💡 **Consumption Rate:** {consumption_per_pair:.4f} m² / pair | Panjang Bahan Terpakai: {max_used_y:.1f} cm dari {sheet_length:.1f} cm")

                st.markdown("---")
                st.subheader("🖼️ ProCost-Style Layout Nesting Result")

                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Canvas Lembaran
                sheet_rect = patches.Rectangle((0, 0), sheet_width, sheet_length, linewidth=2, edgecolor='black', facecolor='#F8F9FA')
                ax.add_patch(sheet_rect)
                
                # Garis Margin Limit
                margin_rect = patches.Rectangle((margin, margin), sheet_width - (2*margin), sheet_length - (2*margin), 
                                                linewidth=1, edgecolor='red', linestyle='--')
                ax.add_patch(margin_rect)

                # Garis Batas Pemotongan Bahan Terpakai
                if max_used_y > 0:
                    ax.axhline(y=max_used_y, color='blue', linestyle=':', linewidth=1.5, label='Actual Cut Line')

                colors = plt.cm.Set3(np.linspace(0, 1, max(len(raw_polygons), 1)))

                for poly, orig_idx in placed:
                    x, y = poly.exterior.xy
                    c = colors[orig_idx % len(colors)]
                    ax.fill(x, y, alpha=0.85, fc=c, ec='black', linewidth=1)

                ax.set_xlim(-5, sheet_width + 5)
                ax.set_ylim(-5, sheet_length + 5)
                ax.set_aspect('equal')
                plt.title(f"Optimized Interlock Layout (Comp. Yield: {component_yield:.1f}% | Pairs: {pairs_completed})", fontsize=12)
                plt.xlabel("Width (cm)")
                plt.ylabel("Length (cm)")
                
                st.pyplot(fig)
