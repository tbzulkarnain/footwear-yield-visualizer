import streamlit as st
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("⚡ Footwear Material Yield Visualizer (Full Control & Real-Time SVG)")
st.caption("Atur Pasangan Master (Pcs 1 & 2) -> Atur Interlock Baris -> Hasil Tampil Instan")

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

# --- RENDERER SVG NATIVE (SUPER FAST) ---
def generate_svg_layout(sheet_w, sheet_l, margin_v, placed_polys, max_y):
    scale = 8
    svg_w = sheet_w * scale
    svg_h = sheet_l * scale

    svg_code = f'<svg width="100%" height="auto" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background-color: #F8F9FA; border: 2px solid #333; border-radius: 8px;">'
    
    m_x = margin_v * scale
    m_y = margin_v * scale
    m_w = (sheet_w - 2 * margin_v) * scale
    m_h = (sheet_l - 2 * margin_v) * scale
    svg_code += f'<rect x="{m_x}" y="{m_y}" width="{m_w}" height="{m_h}" fill="none" stroke="#ff4444" stroke-dasharray="4" stroke-width="1.5"/>'

    if max_y > 0:
        c_y = max_y * scale
        svg_code += f'<line x1="0" y1="{c_y}" x2="{svg_w}" y2="{c_y}" stroke="#3388ff" stroke-dasharray="3" stroke-width="2"/>'

    colors = ['#3388ff', '#ff4444']
    for poly, idx in placed_polys:
        pts = list(poly.exterior.coords)
        pts_str = " ".join([f"{p[0]*scale:.1f},{p[1]*scale:.1f}" for p in pts])
        fill_col = colors[idx % 2]
        svg_code += f'<polygon points="{pts_str}" fill="{fill_col}" stroke="#111" stroke-width="0.8" opacity="0.85"/>'

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

        # ==========================================
        # STEP 1: ATUR PASANGAN MASTER (KOMPONEN 1 & 2)
        # ==========================================
        st.markdown("---")
        st.subheader("🛠️ Step 1: Atur Pasangan Master (Unit Pair)")

        col_c1, col_c2 = st.columns(2)

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
            shift_y2 = st.slider("Geser Y Pcs 2 (cm)", -float(bh), float(bh * 1.5), float(bh * 0.2), step=0.1, key="sy2")

            p2_rot = rotate(base_poly, rot2, origin='center')
            minx2, miny2, _, _ = p2_rot.bounds
            p2_zero = translate(p2_rot, xoff=-minx2, yoff=-miny2)
            poly2_custom = translate(p2_zero, xoff=shift_x2, yoff=shift_y2)

        # Hitung Normalisasi Unit Pair
        p_minx = min(poly1_custom.bounds[0], poly2_custom.bounds[0])
        p_miny = min(poly1_custom.bounds[1], poly2_custom.bounds[1])
        p_maxx = max(poly1_custom.bounds[2], poly2_custom.bounds[2])
        p_maxy = max(poly1_custom.bounds[3], poly2_custom.bounds[3])

        unit_w = p_maxx - p_minx
        unit_h = p_maxy - p_miny

        poly1_unit = translate(poly1_custom, xoff=-p_minx, yoff=-p_miny)
        poly2_unit = translate(poly2_custom, xoff=-p_minx, yoff=-p_miny)

        # Cek Tabrakan Komponen 1 & 2 Sendiri
        pair_collision = poly1_unit.buffer(inter_gap / 2).intersects(poly2_unit.buffer(inter_gap / 2))
        if pair_collision:
            st.warning("⚠️ Komponen 1 dan Komponen 2 di Step 1 saling bertabrakan! Geser slider Komponen 2 agar tidak menindih.")

        # ==========================================
        # STEP 2: ATUR INTERLOCK ANTAR-BARIS
        # ==========================================
        st.markdown("---")
        st.subheader("🚀 Step 2: Atur Interlock Antar-Baris")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            r2_shift_x = st.slider("↔️ Pergeseran Horizontal Baris Genap (cm)", 
                                  min_value=-float(unit_w), 
                                  max_value=float(unit_w), 
                                  value=float(unit_w * 0.5), 
                                  step=0.1)
        with col_s2:
            r2_shift_y = st.slider("↕️ Jarak Vertikal Antar-Baris (cm)", 
                                  min_value=float(unit_h * 0.2), 
                                  max_value=float(unit_h * 1.5), 
                                  value=float(unit_h * 0.8), 
                                  step=0.1)

        # DETEKSI TABRAKAN REAL-TIME BARIS 1, 2, 3
        u1_buf = poly1_unit.buffer(inter_gap / 2)
        u2_buf = poly2_unit.buffer(inter_gap / 2)

        r2_u1 = translate(u1_buf, xoff=r2_shift_x, yoff=r2_shift_y)
        r2_u2 = translate(u2_buf, xoff=r2_shift_x, yoff=r2_shift_y)
        r3_u1 = translate(u1_buf, xoff=0, yoff=r2_shift_y * 2)
        r3_u2 = translate(u2_buf, xoff=0, yoff=r2_shift_y * 2)

        collide_12 = r2_u1.intersects(u1_buf) or r2_u1.intersects(u2_buf) or r2_u2.intersects(u1_buf) or r2_u2.intersects(u2_buf)
        collide_23 = r3_u1.intersects(r2_u1) or r3_u1.intersects(r2_u2) or r3_u2.intersects(r2_u1) or r3_u2.intersects(r2_u2)

        if pair_collision or collide_12 or collide_23:
            st.error("⚠️ POLA BERTAGRAKAN! Geser slider 'Jarak Vertikal' atau atur ulang Posisi Komponen 2.")
        else:
            st.success("✅ LAYOUT SAFE & BEBAS TABRAKAN!")

        # GENERATE LAYOUT FULL SHEET
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

        # METRIK & SUMMARY
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

        # RENDER SVG VISUAL INSTAN
        svg_html = generate_svg_layout(sheet_width, sheet_length, margin, placed_polygons, max_used_y)
        st.components.v1.html(svg_html, height=650, scrolling=True)
