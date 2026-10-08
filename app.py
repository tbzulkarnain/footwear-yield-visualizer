import streamlit as st
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

st.set_page_config(
    page_title="Footwear Material Yield Visualizer",
    page_icon="📐",
    layout="wide"
)

st.title("⚡ Footwear Material Yield Visualizer (ProCost Grid Standard)")
st.caption("Penataan ProCost Otomatis dengan Tambahan Allowance Jarak Potong (mm)")

# ============================================================
# SIDEBAR PARAMETER
# ============================================================

st.sidebar.header("⚙️ Parameter Lembaran Material")

sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin_cm = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
target_pairs = st.sidebar.number_input("Jumlah Pasang Target (Pairs)", value=50, min_value=1, step=1)

# ============================================================
# EXTRACT POLYGONS FROM IMAGE
# ============================================================

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
            if 300 < area_px <= max_area_px:
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


def generate_svg_preview_pair(p1, p2, width_cm=40, height_cm=30, show_p2=True):
    scale = 10
    svg_w = width_cm * scale
    svg_h = height_cm * scale

    svg_code = f'<svg width="100%" height="auto" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#F8F9FA; border:2px dashed #666; border-radius:8px;">'
    items = [(p1, "#3388ff")]
    if show_p2:
        items.append((p2, "#ff4444"))

    for poly, col in items:
        pts = list(poly.exterior.coords)
        pts_str = " ".join([f"{p[0] * scale:.1f},{p[1] * scale:.1f}" for p in pts])
        svg_code += f'<polygon points="{pts_str}" fill="{col}" stroke="#111" stroke-width="1.5" opacity="0.85"/>'

    svg_code += "</svg>"
    return svg_code

# ============================================================
# MAIN APP LOGIC
# ============================================================

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

        # STEP 1: ATUR POSISI & PILIH KATEGORI
        st.markdown("---")
        st.subheader("🛠️ Step 1: Atur Rotasi & Input Distance Allowance (mm)")

        col_ctrl, col_prev = st.columns([1.1, 0.9])

        with col_ctrl:
            st.markdown("##### ⚙️ Rotasi & Kategori ProCost")
            rot_p1 = st.slider("Rotasi Pcs 1 / Biru (°)", 0, 360, 90, step=5)
            
            category = st.selectbox(
                "Pilih Kategori ProCost",
                [
                    "Category 5: Pair Parallel (Pasangan Utuh Lurus)",
                    "Category 6: Pair Staggered (Pasangan Utuh Zig-Zag)",
                    "Category 1: One Way Straight (1 Arah Lurus)",
                    "Category 2: Two Way Interlock (2 Arah 180°)",
                    "Category 3: One Way Staggered (1 Arah Zig-Zag Baris)",
                    "Category 4: Two Way Staggered (2 Arah Zig-Zag Baris)"
                ]
            )

            is_pair_cat = "Pair" in category
            is_twoway_cat = "Two Way" in category or "Category 2" in category or "Category 4" in category

            if is_pair_cat or is_twoway_cat:
                default_rot2 = 270 if is_pair_cat else (rot_p1 + 180) % 360
                rot_p2 = st.slider("Rotasi Pcs 2 / Merah (°)", 0, 360, int(default_rot2), step=5)
            else:
                rot_p2 = rot_p1

            st.markdown("##### 📏 Parameter Jarak Potong (ProCost Allowance)")
            c_gap1, c_gap2 = st.columns(2)
            with c_gap1:
                inter_gap_mm = st.number_input("Allowance Antar-Pola / Gap (mm)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
            with c_gap2:
                pair_gap_mm = st.number_input("Tumpang Tindih Pasangan X (mm)", min_value=-50.0, max_value=50.0, value=5.0, step=0.5)

            c_dist1, c_dist2 = st.columns(2)
            with c_dist1:
                step_x_allowance_mm = st.number_input("Allowance Antar-Pasangan Horiz. X (mm)", min_value=-50.0, max_value=50.0, value=2.0, step=0.5)
            with c_dist2:
                pitch_y_allowance_mm = st.number_input("Allowance Antar-Baris Vert. Y (mm)", min_value=-50.0, max_value=50.0, value=2.0, step=0.5)

        # KONVERSI MM KE CM
        inter_gap_cm = inter_gap_mm / 10.0
        pair_gap_cm = pair_gap_mm / 10.0
        step_x_allowance_cm = step_x_allowance_mm / 10.0
        pitch_y_allowance_cm = pitch_y_allowance_mm / 10.0

        # GEOMETRI P1 & P2
        p1 = rotate(base_poly, rot_p1, origin='center')
        p1 = translate(p1, xoff=-p1.bounds[0], yoff=-p1.bounds[1])

        p2 = rotate(base_poly, rot_p2, origin='center')
        p2 = translate(p2, xoff=-p2.bounds[0], yoff=-p2.bounds[1])
        
        # Posisi P2 relatif terhadap P1
        p2_x_pos = (p1.bounds[2] / 2) + pair_gap_cm
        p2 = translate(p2, xoff=p2_x_pos, yoff=0.0)

        # PREVIEW MASTER PASANGAN
        prev_w = max(p1.bounds[2], p2.bounds[2]) + 5.0
        prev_h = max(p1.bounds[3], p2.bounds[3]) + 5.0

        with col_prev:
            st.markdown("##### 👁️ Preview Master Layout")
            if is_pair_cat or is_twoway_cat:
                if p1.buffer(inter_gap_cm/2).intersects(p2.buffer(inter_gap_cm/2)):
                    st.error("⚠️ Pasangan bertabrakan! Tambahkan nilai 'Tumpang Tindih Pasangan X (mm)'.")
                else:
                    st.success("✅ Jarak Pasangan Aman (Bebas Tabrakan)")

            svg_preview = generate_svg_preview_pair(p1, p2, width_cm=max(prev_w, 25), height_cm=max(prev_h, 20), show_p2=(is_pair_cat or is_twoway_cat))
            st.components.v1.html(svg_preview, height=280, scrolling=False)

        # STEP 2: DUPLIKASI KE LEMBARAN UTUH
        st.markdown("---")
        st.subheader("🚀 Step 2: Duplikasi Ke Lembaran Utuh")

        if st.button("📊 Render Layout ProCost", type="primary"):
            placed_polygons = []
            total_pattern_area = 0.0

            total_items = target_pairs * 2
            item_idx = 0
            row_idx = 0

            # CALCULATE REAL BOUNDING BOX OF UNIT PAIR
            pair_width = max(p1.bounds[2], p2.bounds[2]) - min(p1.bounds[0], p2.bounds[0])
            pair_height = max(p1.bounds[3], p2.bounds[3]) - min(p1.bounds[1], p2.bounds[1])

            # RUMUS MATRIKS PROCOST PRESISI
            step_x = pair_width + inter_gap_cm + step_x_allowance_cm
            pitch_y = pair_height + inter_gap_cm + pitch_y_allowance_cm
            stagger_x = (step_x / 2) if ("Staggered" in category or "Category 3" in category or "Category 4" in category or "Category 6" in category) else 0.0

            while item_idx < total_items:
                is_row_even = (row_idx % 2 == 1)
                row_y = margin_cm + (row_idx * pitch_y)

                if row_y + pair_height > (sheet_length - margin_cm):
                    break

                x_shift = stagger_x if is_row_even else 0.0
                row_start_x = margin_cm + x_shift

                while row_start_x - step_x >= margin_cm:
                    row_start_x -= step_x

                curr_x = row_start_x

                while item_idx < total_items and curr_x <= (sheet_width - margin_cm):
                    
                    if "Category 1" in category or "Category 3" in category:
                        cand_p1 = translate(p1, xoff=curr_x, yoff=row_y)
                        if (cand_p1.bounds[0] >= margin_cm and cand_p1.bounds[2] <= sheet_width - margin_cm and 
                            cand_p1.bounds[1] >= margin_cm and cand_p1.bounds[3] <= sheet_length - margin_cm):
                            placed_polygons.append((cand_p1, 0))
                            total_pattern_area += cand_p1.area
                            item_idx += 1
                    else:
                        cand_p1 = translate(p1, xoff=curr_x, yoff=row_y)
                        cand_p2 = translate(p2, xoff=curr_x, yoff=row_y)

                        p1_in = (cand_p1.bounds[0] >= margin_cm and cand_p1.bounds[2] <= sheet_width - margin_cm and 
                                 cand_p1.bounds[1] >= margin_cm and cand_p1.bounds[3] <= sheet_length - margin_cm)
                        p2_in = (cand_p2.bounds[0] >= margin_cm and cand_p2.bounds[2] <= sheet_width - margin_cm and 
                                 cand_p2.bounds[1] >= margin_cm and cand_p2.bounds[3] <= sheet_length - margin_cm)

                        if p1_in:
                            placed_polygons.append((cand_p1, 0))
                            total_pattern_area += cand_p1.area
                            item_idx += 1

                        if item_idx < total_items and p2_in:
                            placed_polygons.append((cand_p2, 1))
                            total_pattern_area += cand_p2.area
                            item_idx += 1

                    curr_x += step_x

                row_idx += 1

            # SUMMARY METRICS
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

            m_x = margin_cm * scale_f
            m_y = margin_cm * scale_f
            m_w = (sheet_width - 2 * margin_cm) * scale_f
            m_h = (sheet_length - 2 * margin_cm) * scale_f
            svg_full += f'<rect x="{m_x}" y="{m_y}" width="{m_w}" height="{m_h}" fill="none" stroke="#ff4444" stroke-dasharray="4" stroke-width="1.5"/>'

            if max_used_y > 0:
                c_y = max_used_y * scale_f
                svg_full += f'<line x1="0" y1="{c_y}" x2="{svg_w_f}" y2="{c_y}" stroke="#3388ff" stroke-dasharray="3" stroke-width="2"/>'

            colors = ['#3388ff', '#ff4444']
            for poly, idx in placed_polygons:
                pts = list(poly.exterior.coords)
                pts_str = " ".join([f"{p[0] * scale_f:.1f},{p[1] * scale_f:.1f}" for p in pts])
                fill_col = colors[idx % 2]
                svg_full += f'<polygon points="{pts_str}" fill="{fill_col}" stroke="#111" stroke-width="0.8" opacity="0.85"/>'

            svg_full += '</svg>'
            st.components.v1.html(svg_full, height=650, scrolling=True)
