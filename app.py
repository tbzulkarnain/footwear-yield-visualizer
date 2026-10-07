import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("📐 Footwear Material Yield Visualizer (2-Row Block Replication)")
st.caption("Step 1: Atur Pasangan Master -> Step 2: Atur Interlock Baris 1-2 & Baris 2-3 -> Step 3: Replikasi Full Sheet")

# --- SIDEBAR PARAMETER SHEET & NESTING MODE ---
st.sidebar.header("⚙️ Parameter Lembaran Material")
sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
inter_gap = st.sidebar.number_input("Jarak Antar Pola / Interlacing Gap (cm)", value=0.5, step=0.1)

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

        # ==========================================
        # STEP 1: SET PASANGAN MASTER (Pcs 1 & Pcs 2)
        # ==========================================
        st.markdown("---")
        st.subheader("🛠️ Step 1: Atur Pasangan Komponen (Base Pair Unit)")
        st.caption("Atur Komponen 1 (Biru) & Komponen 2 (Merah) hingga membentuk 1 pasang unit yang saling mengunci.")

        col_c1, col_c2, col_preview = st.columns([1, 1, 1.2])

        with col_c1:
            st.markdown("### 🔵 Komponen 1 (Biru)")
            rot1 = st.slider("Rotasi Pcs 1 (°)", 0, 360, 0, step=5, key="rot1")
            shift_x1 = st.slider("Geser X Pcs 1 (cm)", -float(bw), float(bw * 1.5), 0.0, step=0.1, key="sx1")
            shift_y1 = st.slider("Geser Y Pcs 1 (cm)", -float(bh), float(bh * 1.5), 0.0, step=0.1, key="sy1")

            p1_rot = rotate(base_poly, rot1, origin='center')
            minx1, miny1, _, _ = p1_rot.bounds
            p1_zero = translate(p1_rot, xoff=-minx1, yoff=-miny1)
            poly1_custom = translate(p1_zero, xoff=shift_x1, yoff=shift_y1)

        with col_c2:
            st.markdown("### 🔴 Komponen 2 (Merah)")
            rot2 = st.slider("Rotasi Pcs 2 (°)", 0, 360, 180, step=5, key="rot2")
            shift_x2 = st.slider("Geser X Pcs 2 (cm)", -float(bw), float(bw * 1.5), float(bw * 0.4), step=0.1, key="sx2")
            shift_y2 = st.slider("Geser Y Pcs 2 (cm)", -float(bh), float(bh * 1.5), float(bh * 0.5), step=0.1, key="sy2")

            p2_rot = rotate(base_poly, rot2, origin='center')
            minx2, miny2, _, _ = p2_rot.bounds
            p2_zero = translate(p2_rot, xoff=-minx2, yoff=-miny2)
            poly2_custom = translate(p2_zero, xoff=shift_x2, yoff=shift_y2)

        has_pair_overlap = poly1_custom.intersects(poly2_custom)
        if has_pair_overlap:
            st.error("❌ Peringatan: Komponen 1 dan Komponen 2 bertabrakan (overlap)!")
        else:
            st.success("✅ Status: Pasangan Master Valid & Presisi.")

        with col_preview:
            fig_p, ax_p = plt.subplots(figsize=(4.5, 4.5))
            x1, y1 = poly1_custom.exterior.xy
            x2, y2 = poly2_custom.exterior.xy

            ax_p.fill(x1, y1, alpha=0.75, fc='#3388ff', ec='black', linewidth=1.5, label='Komponen 1')
            ax_p.fill(x2, y2, alpha=0.75, fc='#ff4444', ec='black', linewidth=1.5, label='Komponen 2')
            ax_p.set_aspect('equal')
            ax_p.legend(loc='upper right')
            plt.title("Base Pair Unit Preview", fontsize=10)
            st.pyplot(fig_p)

        # Normalisasi Unit Pasangan
        p_minx = min(poly1_custom.bounds[0], poly2_custom.bounds[0])
        p_miny = min(poly1_custom.bounds[1], poly2_custom.bounds[1])
        p_maxx = max(poly1_custom.bounds[2], poly2_custom.bounds[2])
        p_maxy = max(poly1_custom.bounds[3], poly2_custom.bounds[3])

        unit_w = p_maxx - p_minx
        unit_h = p_maxy - p_miny

        poly1_unit = translate(poly1_custom, xoff=-p_minx, yoff=-p_miny)
        poly2_unit = translate(poly2_custom, xoff=-p_minx, yoff=-p_miny)

        # ==========================================
        # STEP 2: SET INTERLOCK BARIS 1-2 & BARIS 2-3
        # ==========================================
        st.markdown("---")
        st.subheader("🛠️ Step 2: Presisikan Hubungan Antar-Baris")
        st.caption("Atur pergeseran Baris 2 terhadap Baris 1, serta jarak batas antara Baris 2 & Baris 3.")

        col_r2_ctrl, col_r2_prev = st.columns([1.1, 1.1])

        with col_r2_ctrl:
            st.markdown("#### 1️⃣ Posisi Baris 2 terhadap Baris 1")
            r2_shift_x = st.slider("↔️ Pergeseran Horizontal Baris 2 (cm)", -float(unit_w), float(unit_w), float(unit_w * 0.25), step=0.1)
            r2_shift_y = st.slider("↕️ Jarak Vertikal Baris 2 dari Baris 1 (cm)", float(unit_h * 0.2), float(unit_h * 1.2), float(unit_h * 0.65), step=0.1)

            st.markdown("#### 2️⃣ Jarak Blok (Baris 2 ke Baris 3)")
            block_gap_y = st.slider("↕️ Jarak Vertikal Baris 3 dari Baris 2 (cm)", float(unit_h * 0.2), float(unit_h * 1.5), float(unit_h * 0.75), step=0.1,
                                    help="Atur nilai ini agar Baris 3 tidak menindih puncak Baris 2.")

            # Buat Pola Baris 1, 2, dan 3 untuk Verifikasi Interlock
            row1_p1 = poly1_unit
            row1_p2 = poly2_unit

            row2_p1 = translate(poly1_unit, xoff=r2_shift_x, yoff=r2_shift_y)
            row2_p2 = translate(poly2_unit, xoff=r2_shift_x, yoff=r2_shift_y)

            row3_p1 = translate(poly1_unit, xoff=0.0, yoff=r2_shift_y + block_gap_y)
            row3_p2 = translate(poly2_unit, xoff=0.0, yoff=r2_shift_y + block_gap_y)

            # Cek Benturan (Collision Checks)
            overlap_12 = (row2_p1.intersects(row1_p1) or row2_p1.intersects(row1_p2) or row2_p2.intersects(row1_p1) or row2_p2.intersects(row1_p2))
            overlap_23 = (row3_p1.intersects(row2_p1) or row3_p1.intersects(row2_p2) or row3_p2.intersects(row2_p1) or row3_p2.intersects(row2_p2))

            if overlap_12:
                st.error("⚠️ Baris 2 bertabrakan dengan Baris 1! Naikkkan 'Jarak Vertikal Baris 2 dari Baris 1'.")
            elif overlap_23:
                st.error("⚠️ Baris 3 bertabrakan dengan Baris 2! Naikkkan 'Jarak Vertikal Baris 3 dari Baris 2'.")
            else:
                st.success("✅ Interlock 3 Baris Pertama (Baris 1, 2, 3) 100% Bebas Tabrakan!")

        with col_r2_prev:
            fig_r3, ax_r3 = plt.subplots(figsize=(6, 5.5))
            
            # Row 1
            ax_r3.fill(*row1_p1.exterior.xy, alpha=0.7, fc='#3388ff', ec='black', label='Baris 1 (Pcs 1)')
            ax_r3.fill(*row1_p2.exterior.xy, alpha=0.7, fc='#ff4444', ec='black', label='Baris 1 (Pcs 2)')
            
            # Row 2
            ax_r3.fill(*row2_p1.exterior.xy, alpha=0.8, fc='#1155bb', ec='black', hatch='//', label='Baris 2 (Pcs 1)')
            ax_r3.fill(*row2_p2.exterior.xy, alpha=0.8, fc='#bb1111', ec='black', hatch='//', label='Baris 2 (Pcs 2)')

            # Row 3
            ax_r3.fill(*row3_p1.exterior.xy, alpha=0.6, fc='#3388ff', ec='black', linestyle='--', label='Baris 3 (Pcs 1)')
            ax_r3.fill(*row3_p2.exterior.xy, alpha=0.6, fc='#ff4444', ec='black', linestyle='--', label='Baris 3 (Pcs 2)')

            ax_r3.set_aspect('equal')
            ax_r3.legend(loc='upper right', fontsize=7)
            plt.title("Preview 3-Row Block Interlock Verification", fontsize=10)
            st.pyplot(fig_r3)

        # ==========================================
        # STEP 3: REPLIKASI OTOMATIS SELURUH LEMBARAN
        # ==========================================
        st.markdown("---")
        st.subheader("🚀 Step 3: Live Replikasi Full Sheet Layout")

        placed_polygons = []
        total_pattern_area = 0.0

        total_items = target_pairs * 2
        item_idx = 0
        row_idx = 0

        # Tinggi 1 Blok Pasangan (Baris Genap + Baris Ganjil)
        pair_block_height = r2_shift_y + block_gap_y

        curr_base_y = margin

        while item_idx < total_items and (curr_base_y + min(poly1_unit.bounds[3], poly2_unit.bounds[3])) <= (sheet_length - margin):
            is_row_even = (row_idx % 2 == 0)
            
            # Hitung offset koordinat Y dan X presisi untuk setiap baris
            block_number = row_idx // 2
            
            if is_row_even:
                row_y = margin + (block_number * pair_block_height)
                x_offset = 0.0
            else:
                row_y = margin + (block_number * pair_block_height) + r2_shift_y
                x_offset = r2_shift_x

            curr_x = margin + x_offset

            while curr_x < margin:
                curr_x += (unit_w + inter_gap)

            while item_idx < total_items and (curr_x + unit_w) <= (sheet_width - margin):
                # Komponen 1 (Biru)
                p1 = translate(poly1_unit, xoff=curr_x, yoff=row_y)
                if p1.bounds[2] <= (sheet_width - margin) and p1.bounds[3] <= (sheet_length - margin) and p1.bounds[0] >= margin:
                    placed_polygons.append((p1, 0))
                    total_pattern_area += p1.area
                    item_idx += 1

                # Komponen 2 (Merah)
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

        # VISUALISASI HASIL FULL SHEET
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
        plt.title(f"3-Row Verified Replicated Layout | Comp. Yield: {component_yield:.1f}% | Pairs: {pairs_completed}", fontsize=12)
        plt.xlabel("Width (cm)")
        plt.ylabel("Length (cm)")
        
        st.pyplot(fig)
