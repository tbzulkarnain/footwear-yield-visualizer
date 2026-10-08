import streamlit as st
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("⚡ Footwear Material Yield Visualizer (3-Component Master Preview)")
st.caption("Atur 3 Komponen Master -> Duplikasi Matriks Presisi 100% Bebas Tabrakan")

# --- SIDEBAR PARAMETER SHEET ---
st.sidebar.header("⚙️ Parameter Lembaran Material")
sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
inter_gap = st.sidebar.number_input("Jarak Antar Pola / Interlacing Gap (cm)", value=0.1, step=0.05)

target_pairs = st.sidebar.number_input("Jumlah Pasang Target (Pairs)", value=50, min_value=1, step=1)

# --- FUNGSI OPENCV EKSTRAKSI KONTUR ---
@st.cache_data
def extract_polygons_from_bytes(file_bytes, dpi=96):
    try:
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
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
    except Exception:
        return []

# --- RENDERER SVG PREVIEW 3 KOMPONEN ---
def generate_svg_3pcs_preview(p1, p2, p3, width_cm=40, height_cm=35):
    scale = 10
    svg_w = width_cm * scale
    svg_h = height_cm * scale

    svg_code = f'<svg width="100%" height="auto" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background-color: #F8F9FA; border: 2px dashed #666; border-radius: 8px;">'
    
    items = [(p1, '#3388ff'), (p2, '#ff4444'), (p3, '#28a745')]
    
    for poly, col in items:
        pts = list(poly.exterior.coords)
        pts_str = " ".join([f"{p[0]*scale:.1f},{p[1]*scale:.1f}" for p in pts])
        svg_code += f'<polygon points="{pts_str}" fill="{col}" stroke="#111" stroke-width="1" opacity="0.85"/>'

    svg_code += '</svg>'
    return svg_code

# --- MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload Gambar Pattern Component Master", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    raw_polygons = extract_polygons_from_bytes(file_bytes)
    
    if not raw_polygons:
        st.error("Gagal mendeteksi bentuk pola dari gambar. Pastikan garis kontur pola jelas.")
    else:
        base_poly = raw_polygons[0]
        bw = base_poly.bounds[2] - base_poly.bounds[0]
        bh = base_poly.bounds[3] - base_poly.bounds[1]

        st.markdown("---")
        st.subheader("🛠️ Step 1: Atur Posisi Master 3 Komponen")

        col_ctrl, col_prev = st.columns([1.1, 0.9])

        with col_ctrl:
            st.markdown("##### 🔵 Komponen 1 (Baris 1 - Pcs 1)")
            c1_1, c1_2, c1_3 = st.columns(3)
            with c1_1:
                rot1 = st.slider("Rotasi Pcs 1 (°)", 0, 360, 0, step=5)
            with c1_2:
                shift_x1 = st.slider("Geser X Pcs 1", -float(bw), float(bw * 1.5), 0.0, step=0.1)
            with c1_3:
                shift_y1 = st.slider("Geser Y Pcs 1", -float(bh), float(bh * 1.5), 0.0, step=0.1)

            st.markdown("##### 🔴 Komponen 2 (Baris 1 - Pcs 2)")
            c2_1, c2_2, c2_3 = st.columns(3)
            with c2_1:
                rot2 = st.slider("Rotasi Pcs 2 (°)", 0, 360, 180, step=5)
            with c2_2:
                shift_x2 = st.slider("Geser X Pcs 2", -float(bw), float(bw * 1.5), float(bw * 0.4), step=0.1)
            with c2_3:
                shift_y2 = st.slider("Geser Y Pcs 2", -float(bh), float(bh * 1.5), float(bh * 0.2), step=0.1)

            st.markdown("##### 🟢 Komponen 3 (Awal Baris 2)")
            c3_1, c3_2, c3_3 = st.columns(3)
            with c3_1:
                rot3 = st.slider("Rotasi Pcs 3 (°)", 0, 360, rot1, step=5)
            with c3_2:
                r2_shift_x = st.slider("Geser X Pcs 3", -float(bw*1.5), float(bw*1.5), float(bw * 0.5), step=0.1)
            with c3_3:
                r2_shift_y = st.slider("Geser Y Pcs 3", float(bh * 0.2), float(bh * 2.0), float(bh * 0.8), step=0.1)

        # GEOMETRI 3 KOMPONEN
        p1_rot = rotate(base_poly, rot1, origin='center')
        minx1, miny1, _, _ = p1_rot.bounds
        poly1_custom = translate(p1_rot, xoff=-minx1 + shift_x1, yoff=-miny1 + shift_y1)

        p2_rot = rotate(base_poly, rot2, origin='center')
        minx2, miny2, _, _ = p2_rot.bounds
        poly2_custom = translate(p2_rot, xoff=-minx2 + shift_x2, yoff=-miny2 + shift_y2)

        p3_rot = rotate(base_poly, rot3, origin='center')
        minx3, miny3, _, _ = p3_rot.bounds
        poly3_custom = translate(p3_rot, xoff=-minx3 + r2_shift_x, yoff=-miny3 + r2_shift_y)

        # Normalisasi Unit Pair Baris 1 ke Titik Nol (0,0)
        p_minx = min(poly1_custom.bounds[0], poly2_custom.bounds[0])
        p_miny = min(poly1_custom.bounds[1], poly2_custom.bounds[1])

        poly1_zero = translate(poly1_custom, xoff=-p_minx, yoff=-p_miny)
        poly2_zero = translate(poly2_custom, xoff=-p_minx, yoff=-p_miny)
        poly3_zero = translate(poly3_custom, xoff=-p_minx, yoff=-p_miny)

        # Lebar Efektif Pasangan (Unit Width & Height)
        unit_w = max(poly1_zero.bounds[2], poly2_zero.bounds[2]) - min(poly1_zero.bounds[0], poly2_zero.bounds[0])
        unit_h = max(poly1_zero.bounds[3], poly2_zero.bounds[3]) - min(poly1_zero.bounds[1], poly2_zero.bounds[1])

        # Vektor Pergeseran Baris 2 relatif terhadap Komponen 1
        row2_offset_x = poly3_zero.bounds[0] - poly1_zero.bounds[0]
        row2_offset_y = poly3_zero.bounds[1] - poly1_zero.bounds[1]

        # Padding Tampilan Preview
        pad = 5.0
        p1_unit = translate(poly1_zero, xoff=pad, yoff=pad)
        p2_unit = translate(poly2_zero, xoff=pad, yoff=pad)
        p3_unit = translate(poly3_zero, xoff=pad, yoff=pad)

        # DETEKSI TABRAKAN REALTIME PREVIEW
        u1_b = p1_unit.buffer(inter_gap / 2)
        u2_b = p2_unit.buffer(inter_gap / 2)
        u3_b = p3_unit.buffer(inter_gap / 2)

        collide_12 = u1_b.intersects(u2_b)
        collide_13 = u1_b.intersects(u3_b)
        collide_23 = u2_b.intersects(u3_b)

        with col_prev:
            st.markdown("### 👁️ Preview Master (3 Komponen)")
            if collide_12 or collide_13 or collide_23:
                st.error("⚠️ Terdapat Komponen yang Bertabrakan! Adjust slider sampai posisi aman.")
            else:
                st.success("✅ 3 Komponen Bebas Tabrakan (Layout Safe)")

            pw = max(p1_unit.bounds[2], p2_unit.bounds[2], p3_unit.bounds[2]) + pad
            ph = max(p1_unit.bounds[3], p2_unit.bounds[3], p3_unit.bounds[3]) + pad
            
            svg_3pcs = generate_svg_3pcs_preview(p1_unit, p2_unit, p3_unit, width_cm=max(pw, 25), height_cm=max(ph, 25))
            st.components.v1.html(svg_3pcs, height=380, scrolling=False)

        # --- STEP 2: DUPLIKASI KE LEMBARAN UTUH ---
        st.markdown("---")
        st.subheader("🚀 Step 2: Duplikasi Ke Lembaran Utuh")

        if st.button("📊 Duplikasi & Render Full Sheet Layout", type="primary"):
            placed_polygons = []
            total_pattern_area = 0.0

            total_items = target_pairs * 2
            item_idx = 0
            row_idx = 0

            curr_base_y = margin

            # Step per x (jarak antar pasangan sejajar horizontal)
            step_x = unit_w + inter_gap

            while item_idx < total_items:
                is_row_even = (row_idx % 2 == 0)
                
                # Hitung Y Baris Berdasarkan Vektor Pergeseran
                row_y = margin + (row_idx * row2_offset_y)
                
                # Pergeseran X Selang-Seling Baris
                x_shift = (row_idx * row2_offset_x)

                # Jika Y melampaui lembaran, hentikan
                if row_y + min(poly1_zero.bounds[3], poly2_zero.bounds[3]) > (sheet_length - margin):
                    break

                curr_x = margin + x_shift

                # Kembalikan x ke area lembaran jika terlalu ke kiri/kanan
                while curr_x < margin:
                    curr_x += step_x

                while item_idx < total_items and (curr_x + unit_w) <= (sheet_width - margin):
                    # Komponen 1 (Biru)
                    p1 = translate(poly1_zero, xoff=curr_x, yoff=row_y)
                    if p1.bounds[2] <= (sheet_width - margin) and p1.bounds[3] <= (sheet_length - margin) and p1.bounds[0] >= margin:
                        placed_polygons.append((p1, 0))
                        total_pattern_area += p1.area
                        item_idx += 1

                    # Komponen 2 (Merah)
                    if item_idx < total_items:
                        p2 = translate(poly2_zero, xoff=curr_x, yoff=row_y)
                        if p2.bounds[2] <= (sheet_width - margin) and p2.bounds[3] <= (sheet_length - margin) and p2.bounds[0] >= margin:
                            placed_polygons.append((p2, 1))
                            total_pattern_area += p2.area
                            item_idx += 1

                    curr_x += step_x

                row_idx += 1

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

            # RENDER SVG FULL SHEET
            scale_f = 8
            svg_w_f = sheet_width * scale_f
            svg_h_f = sheet_length * scale_f

            svg_full = f'<svg width="100%" height="auto" viewBox="0 0 {svg_w_f} {svg_h_f}" xmlns="http://www.w3.org/2000/svg" style="background-color: #F8F9FA; border: 2px solid #333; border-radius: 8px;">'
            
            m_x = margin * scale_f
            m_y = margin * scale_f
            m_w = (sheet_width - 2 * margin) * scale_f
            m_h = (sheet_length - 2 * margin) * scale_f
            svg_full += f'<rect x="{m_x}" y="{m_y}" width="{m_w}" height="{m_h}" fill="none" stroke="#ff4444" stroke-dasharray="4" stroke-width="1.5"/>'

            if max_used_y > 0:
                c_y = max_used_y * scale_f
                svg_full += f'<line x1="0" y1="{c_y}" x2="{svg_w_f}" y2="{c_y}" stroke="#3388ff" stroke-dasharray="3" stroke-width="2"/>'

            colors = ['#3388ff', '#ff4444']
            for poly, idx in placed_polygons:
                pts = list(poly.exterior.coords)
                pts_str = " ".join([f"{p[0]*scale_f:.1f},{p[1]*scale_f:.1f}" for p in pts])
                fill_col = colors[idx % 2]
                svg_full += f'<polygon points="{pts_str}" fill="{fill_col}" stroke="#111" stroke-width="0.8" opacity="0.85"/>'

            svg_full += '</svg>'
            st.components.v1.html(svg_full, height=650, scrolling=True)
