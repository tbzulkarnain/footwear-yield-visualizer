import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("⚡ Footwear Material Yield Visualizer (Instan & Bebas Tabrakan)")
st.caption("Atur Posisi Pasangan & Jarak Baris Aman secara Real-Time tanpa Waiting Time")

# --- SIDEBAR PARAMETER SHEET ---
st.sidebar.header("⚙️ Parameter Lembaran Material")
sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
inter_gap = st.sidebar.number_input("Jarak Antar Pola / Interlacing Gap (cm)", value=0.1, step=0.05)

target_pairs = st.sidebar.number_input("Jumlah Pasang Target (Pairs)", value=50, min_value=1, step=1)

# --- FUNGSI OPENCV UNTUK AMBIL POLYGON POLA ---
def extract_polygons_from_image(uploaded_file, dpi=96):
    try:
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
    except Exception as e:
        st.error(f"Error pembacaan gambar: {e}")
        return []

# --- UPLOAD GAMBAR ---
uploaded_file = st.file_uploader("Upload Gambar Pattern Component Master", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    raw_polygons = extract_polygons_from_image(uploaded_file)
    
    if not raw_polygons:
        st.error("Gagal mendeteksi bentuk pola dari gambar. Pastikan kontur garis pola jelas.")
    else:
        base_poly = raw_polygons[0]
        bw = base_poly.bounds[2] - base_poly.bounds[0]
        bh = base_poly.bounds[3] - base_poly.bounds[1]

        st.markdown("---")
        st.subheader("🛠️ Step 1: Atur Pasangan Master (Unit Pair)")

        col_c1, col_c2, col_prev = st.columns([1, 1, 1.2])

        with col_c1:
            st.markdown("### 🔵 Komponen 1 (Biru)")
            rot1 = st.slider("Rotasi Pcs 1 (°)", 0, 360, 0, step=5, key="r1")
            shift_x1 = st.slider("Geser X Pcs 1 (cm)", -float(bw), float(bw * 1.5), 0.0, step=0.1, key="sx1")
            shift_y1 = st.slider("Geser Y Pcs 1 (cm)", -float(bh), float(bh * 1.5), 0.0, step=0.1, key="sy1")

            p1_rot = rotate(base_poly, rot1, origin='center')
            minx1, miny1, _, _ = p1_rot.bounds
            p1_zero = translate(p1_rot, xoff=-minx1, yoff=-miny1)
            poly1_custom = translate(p1_zero, xoff=shift_x1, yoff=shift_y1)

        with col_c2:
            st.markdown("### 🔴 Komponen 2 (Merah)")
            rot2 = st.slider("Rotasi Pcs 2 (°)", 0, 360, 180, step=5, key="r2")
            shift_x2 = st.slider("Geser X Pcs 2 (cm)", -float(bw), float(bw * 1.5), float(bw * 0.4), step=0.1, key="sx2")
            shift_y2 = st.slider("Geser Y Pcs 2 (cm)", -float(bh), float(bh * 1.5), float(bh * 0.5), step=0.1, key="sy2")

            p2_rot = rotate(base_poly, rot2, origin='center')
            minx2, miny2, _, _ = p2_rot.bounds
            p2_zero = translate(p2_rot, xoff=-minx2, yoff=-miny2)
            poly2_custom = translate(p2_zero, xoff=shift_x2, yoff=shift_y2)

        # Hitung Normalisasi Pair
        p_minx = min(poly1_custom.bounds[0], poly2_custom.bounds[0])
        p_miny = min(poly1_custom.bounds[1], poly2_custom.bounds[1])
        p_maxx = max(poly1_custom.bounds[2], poly2_custom.bounds[2])
        p_maxy = max(poly1_custom.bounds[3], poly2_custom.bounds[3])

        unit_w = p_maxx - p_minx
        unit_h = p_maxy - p_miny

        poly1_unit = translate(poly1_custom, xoff=-p_minx, yoff=-p_miny)
        poly2_unit = translate(poly2_custom, xoff=-p_minx, yoff=-p_miny)

        with col_prev:
            fig_p, ax_p = plt.subplots(figsize=(4.5, 4.5))
            ax_p.fill(*poly1_unit.exterior.xy, alpha=0.75, fc='#3388ff', ec='black', linewidth=1.5, label='Komponen 1')
            ax_p.fill(*poly2_unit.exterior.xy, alpha=0.75, fc='#ff4444', ec='black', linewidth=1.5, label='Komponen 2')
            ax_p.set_aspect('equal')
            ax_p.legend(loc='upper right')
            plt.title("Preview Master Pair", fontsize=10)
            st.pyplot(fig_p)

        st.markdown("---")
        st.subheader("🚀 Step 2: Atur Interlock Antar-Baris (Live Update)")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            r2_shift_x = st.slider("↔️ Pergeseran Horizontal Baris 2 (cm)", -float(unit_w), float(unit_w), float(unit_w * 0.25), step=0.1)
        with col_r2:
            r2_shift_y = st.slider("↕️ Jarak Vertikal Antar-Baris (cm)", float(unit_h * 0.3), float(unit_h * 1.2), float(unit_h * 0.7), step=0.1)

        # DETEKSI TABRAKAN SEPENUHNYA INSTAN SAAT SLIDER DIGESER
        r2_p1 = translate(poly1_unit, xoff=r2_shift_x, yoff=r2_shift_y)
        r2_p2 = translate(poly2_unit, xoff=r2_shift_x, yoff=r2_shift_y)

        has_collision = (
            r2_p1.intersects(poly1_unit) or r2_p1.intersects(poly2_unit) or
            r2_p2.intersects(poly1_unit) or r2_p2.intersects(poly2_unit)
        )

        if has_collision:
            st.error("⚠️ Posisi Baris 2 bertabrakan dengan Baris 1! Naikkkan nilai 'Jarak Vertikal Antar-Baris'.")
        else:
            st.success("✅ Layout 100% Bebas Tabrakan!")

        # GENERATE LAYOUT INSTAN
        placed_polygons = []
        total_pattern_area = 0.0

        total_items = target_pairs * 2
        item_idx = 0
        row_idx = 0

        curr_base_y = margin

        while item_idx < total_items and (curr_base_y + min(poly1_unit.bounds[3], poly2_unit.bounds[3])) <= (sheet_length - margin):
            is_row_even = (row_idx % 2 == 0)
            
            row_y = margin + (row_idx * r2_shift_y)
            x_offset = r2_shift_x if not is_row_even else 0.0

            curr_x = margin + x_offset

            while curr_x < margin:
                curr_x += (unit_w + inter_gap)

            while item_idx < total_items and (curr_x + unit_w) <= (sheet_width - margin):
                p1 = translate(poly1_unit, xoff=curr_x, yoff=row_y)
                if p1.bounds[2] <= (sheet_width - margin) and p1.bounds[3] <= (sheet_length - margin) and p1.bounds[0] >= margin:
                    placed_polygons.append((p1, 0))
                    total_pattern_area += p1.area
                    item_idx += 1

                if item_idx < total_items:
                    p2 = translate(poly2_unit, xoff=curr_x, yoff=row_y)
                    if p2.bounds[2] <= (sheet_width - margin) and p2.bounds[3] <= (sheet_length - margin) and p2.bounds[0] >= margin:
                        placed_polygons.append((p2, 1))
                        total_pattern_area += p2.area
                        item_idx += 1

                curr_x += unit_w + inter_gap

            row_idx += 1

        # METRIK SUMMARY
        total_sheet_area = sheet_width * sheet_length
        max_used_y = max([p.bounds[3] for p, _ in placed_polygons]) if placed_polygons else 0.0
        used_sheet_area = sheet_width * max_used_y if max_used_y > 0 else total_sheet_area

        component_yield = (total_pattern_area / used_sheet_area) * 100 if used_sheet_area > 0 else 0.0
        overall_sheet_yield = (total_pattern_area / total_sheet_area) * 100
        total_waste = 100.0 - component_yield
        
        pairs_completed = len(placed_polygons) // 2
        consumption_per_pair = (used_sheet_area / 10000) / max(pairs_completed, 1)

        st.markdown("### 📊 Yield & Material Consumption Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Komponen Terpasang", f"{len(placed_polygons)} pcs ({pairs_completed} pairs)")
        m2.metric("Total Net Area", f"{total_pattern_area:.1f} cm²")
        m3.metric("Component Yield", f"{component_yield:.2f} %")
        m4.metric("Overall Sheet Yield", f"{overall_sheet_yield:.2f} %")
        m5.metric("Cutting Waste", f"{total_waste:.2f} %")

        st.info(f"💡 **Consumption Rate:** {consumption_per_pair:.4f} m² / pair | Panjang Bahan Terpakai: {max_used_y:.1f} cm dari {sheet_length:.1f} cm")

        # VISUALISASI FULL SHEET
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sheet_rect = patches.Rectangle((0, 0), sheet_width, sheet_length, linewidth=2, edgecolor='black', facecolor='#F8F9FA')
        ax.add_patch(sheet_rect)
        
        margin_rect = patches.Rectangle((margin, margin), sheet_width - (2*margin), sheet_length - (2*margin), 
                                        linewidth=1, edgecolor='red', linestyle='--')
        ax.add_patch(margin_rect)

        if max_used_y > 0:
            ax.axhline(y=max_used_y, color='blue', linestyle=':', linewidth=1.5, label='Actual Cut Line')

        colors = ['#3388ff', '#ff4444']

        for poly, idx in placed_polygons:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.85, fc=colors[idx % 2], ec='black', linewidth=1)

        ax.set_xlim(-5, sheet_width + 5)
        ax.set_ylim(-5, sheet_length + 5)
        ax.set_aspect('equal')
        plt.title(f"Instant Real-Time Layout | Comp. Yield: {component_yield:.1f}% | Pairs: {pairs_completed}", fontsize=12)
        plt.xlabel("Width (cm)")
        plt.ylabel("Length (cm)")
        
        st.pyplot(fig)
