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

st.title("⚡ Footwear Material Yield Visualizer (Dynamic 6 ProCost Categories)")
st.caption("Kategori Layout ProCost Berbeda Sesuai Aturan Masing-Masing")

# ============================================================
# SIDEBAR PARAMETER
# ============================================================

st.sidebar.header("⚙️ Parameter Lembaran Material")

sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
inter_gap = st.sidebar.number_input("Jarak Antar Pola / Interlacing Gap (cm)", value=0.1, step=0.05)
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

        # STEP 1: KONTROL ROTASI & KATEGORI
        st.markdown("---")
        st.subheader("🛠️ Step 1: Atur Posisi & Pilih Kategori Layout")

        col_ctrl, col_prev = st.columns([1.1, 0.9])

        with col_ctrl:
            st.markdown("##### ⚙️ Kontrol Rotasi & Kategori ProCost")
            rot_p1 = st.slider("Rotasi Pcs 1 / Biru (°)", 0, 360, 90, step=5)
            
            category = st.selectbox(
                "Pilih Kategori ProCost",
                [
                    "Category 1: One Way Straight (1 Arah Lurus)",
                    "Category 2: Two Way Interlock (2 Arah 180°)",
                    "Category 3: One Way Staggered (1 Arah Zig-Zag Baris)",
                    "Category 4: Two Way Staggered (2 Arah Zig-Zag Baris)",
                    "Category 5: Pair Parallel (Pasangan Utuh Lurus)",
                    "Category 6: Pair Staggered (Pasangan Utuh Zig-Zag)"
                ]
            )

            is_pair_cat = "Pair" in category
            is_twoway_cat = "Two Way" in category

            if is_pair_cat or is_twoway_cat:
                default_rot2 = (rot_p1 + 180) % 360 if is_twoway_cat else 270
                rot_p2 = st.slider("Rotasi Pcs 2 / Merah (°)", 0, 360, int(default_rot2), step=5)
            else:
                rot_p2 = rot_p1

            if is_pair_cat:
                pair_offset_x = st.slider("Atur Jarak X Pcs 2 (Merah ke Biru)", 0.0, float(bw * 2), float(bw * 0.7), step=0.1)
            else:
                pair_offset_x = 0.0

        # HITUNG GEOMETRI P1 & P2
        p1 = rotate(base_poly, rot_p1, origin='center')
        p1 = translate(p1, xoff=-p1.bounds[0], yoff=-p1.bounds[1])

        p2 = rotate(base_poly, rot_p2, origin='center')
        p2 = translate(p2, xoff=-p2.bounds[0], yoff=-p2.bounds[1])
        if is_pair_cat:
            p2 = translate(p2, xoff=pair_offset_x, yoff=0.0)

        # PREVIEW MASTER
        prev_w = max(p1.bounds[2], p2.bounds[2]) + 5.0
        prev_h = max(p1.bounds[3], p2.bounds[3]) + 5.0

        with col_prev:
            st.markdown("##### 👁️ Preview Master Layout")
            if is_pair_cat:
                if p1.buffer(inter_gap/2).intersects(p2.buffer(inter_gap/2)):
                    st.error("⚠️ Pasangan bertabrakan! Geser slider 'Jarak X Pcs 2' ke kanan.")
                else:
                    st.success("✅ Jarak Pasangan Aman")
            else:
                st.info(f"💡 Layout Mode: **{category.split(':')[0]}**")

            svg_preview = generate_svg_preview_pair(p1, p2, width_cm=max(prev_w, 25), height_cm=max(prev_h, 20), show_p2=(is_pair_cat or is_twoway_cat))
            st.components.v1.html(svg_preview, height=280, scrolling=False)

        # STEP 2: DUPLIKASI SPESIFIK SESUAI KATEGORI
        st.markdown("---")
        st.subheader("🚀 Step 2: Duplikasi Ke Lembaran Utuh")

        if st.button("📊 Render Layout ProCost", type="primary"):
            placed_polygons = []
            total_pattern_area = 0.0

            total_items = target_pairs * 2
            item_idx = 0
            row_idx = 0

            # ----------------------------------------------------
            # ATURAN MINGGIR & PITCH BERDASARKAN KATEGORI
            # ----------------------------------------------------

            if "Category 1" in category:
                # 1-Way Straight: Murni P1 berurutan
                step_x = p1.bounds[2] + inter_gap
                pitch_y = p1.bounds[3] + inter_gap
                
                while item_idx < total_items:
                    row_y = margin + (row_idx * pitch_y)
                    if row_y + p1.bounds[3] > (sheet_length - margin):
                        break
                    
                    curr_x = margin
                    while item_idx < total_items and (curr_x + p1.bounds[2]) <= (sheet_width - margin):
                        cand = translate(p1, xoff=curr_x, yoff=row_y)
                        placed_polygons.append((cand, 0))
                        total_pattern_area += cand.area
                        item_idx += 1
                        curr_x += step_x

                    row_idx += 1

            elif "Category 2" in category:
                # 2-Way Interlock: P1 (0°) & P2 (180°) Selang-Seling Horizontal
                w_unit = max(p1.bounds[2], p2.bounds[2])
                step_x = w_unit + inter_gap
                pitch_y = max(p1.bounds[3], p2.bounds[3]) + inter_gap

                while item_idx < total_items:
                    row_y = margin + (row_idx * pitch_y)
                    if row_y + max(p1.bounds[3], p2.bounds[3]) > (sheet_length - margin):
                        break
                    
                    curr_x = margin
                    col_idx = 0
                    while item_idx < total_items and (curr_x + w_unit) <= (sheet_width - margin):
                        p_curr = p1 if col_idx % 2 == 0 else p2
                        color_idx = 0 if col_idx % 2 == 0 else 1
                        cand = translate(p_curr, xoff=curr_x, yoff=row_y)
                        placed_polygons.append((cand, color_idx))
                        total_pattern_area += cand.area
                        item_idx += 1
                        curr_x += step_x
                        col_idx += 1

                    row_idx += 1

            elif "Category 3" in category:
                # 1-Way Staggered: P1 Murni dengan Baris Genap Geser Horizontal
                step_x = p1.bounds[2] + inter_gap
                pitch_y = p1.bounds[3] + inter_gap
                stagger_x = step_x / 2

                while item_idx < total_items:
                    is_row_even = (row_idx % 2 == 1)
                    row_y = margin + (row_idx * pitch_y)
                    if row_y + p1.bounds[3] > (sheet_length - margin):
                        break
                    
                    row_start_x = margin + (stagger_x if is_row_even else 0.0)
                    while row_start_x - step_x >= margin:
                        row_start_x -= step_x

                    curr_x = row_start_x
                    while item_idx < total_items and curr_x <= (sheet_width - margin):
                        if curr_x >= margin and (curr_x + p1.bounds[2]) <= (sheet_width - margin):
                            cand = translate(p1, xoff=curr_x, yoff=row_y)
                            placed_polygons.append((cand, 0))
                            total_pattern_area += cand.area
                            item_idx += 1
                        curr_x += step_x

                    row_idx += 1

            elif "Category 4" in category:
                # 2-Way Staggered: P1 & P2 Selang-seling + Baris Genap Geser Horizontal
                w_unit = max(p1.bounds[2], p2.bounds[2])
                step_x = w_unit + inter_gap
                pitch_y = max(p1.bounds[3], p2.bounds[3]) + inter_gap
                stagger_x = step_x / 2

                while item_idx < total_items:
                    is_row_even = (row_idx % 2 == 1)
                    row_y = margin + (row_idx * pitch_y)
                    if row_y + max(p1.bounds[3], p2.bounds[3]) > (sheet_length - margin):
                        break
                    
                    row_start_x = margin + (stagger_x if is_row_even else 0.0)
                    while row_start_x - step_x >= margin:
                        row_start_x -= step_x

                    curr_x = row_start_x
                    col_idx = 0
                    while item_idx < total_items and curr_x <= (sheet_width - margin):
                        if curr_x >= margin and (curr_x + w_unit) <= (sheet_width - margin):
                            p_curr = p1 if col_idx % 2 == 0 else p2
                            color_idx = 0 if col_idx % 2 == 0 else 1
                            cand = translate(p_curr, xoff=curr_x, yoff=row_y)
                            placed_polygons.append((cand, color_idx))
                            total_pattern_area += cand.area
                            item_idx += 1
                        curr_x += step_x
                        col_idx += 1

                    row_idx += 1

            else:
                # Category 5 & 6: Pair Unit (Blok Pasangan P1 + P2)
                pair_width = max(p1.bounds[2], p2.bounds[2])
                pair_height = max(p1.bounds[3], p2.bounds[3])

                step_x = pair_width + inter_gap
                pitch_y = pair_height + inter_gap
                stagger_x = (step_x / 2) if "Category 6" in category else 0.0

                while item_idx < total_items:
                    is_row_even = (row_idx % 2 == 1)
                    row_y = margin + (row_idx * pitch_y)

                    if row_y + pair_height > (sheet_length - margin):
                        break

                    x_shift = stagger_x if is_row_even else 0.0
                    row_start_x = margin + x_shift

                    while row_start_x - step_x >= margin:
                        row_start_x -= step_x

                    curr_x = row_start_x

                    while item_idx < total_items and curr_x <= (sheet_width - margin):
                        cand_p1 = translate(p1, xoff=curr_x, yoff=row_y)
                        cand_p2 = translate(p2, xoff=curr_x, yoff=row_y)

                        p1_in = (cand_p1.bounds[0] >= margin and cand_p1.bounds[2] <= sheet_width - margin and 
                                 cand_p1.bounds[1] >= margin and cand_p1.bounds[3] <= sheet_length - margin)
                        p2_in = (cand_p2.bounds[0] >= margin and cand_p2.bounds[2] <= sheet_width - margin and 
                                 cand_p2.bounds[1] >= margin and cand_p2.bounds[3] <= sheet_length - margin)

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
                pts_str = " ".join([f"{p[0] * scale_f:.1f},{p[1] * scale_f:.1f}" for p in pts])
                fill_col = colors[idx % 2]
                svg_full += f'<polygon points="{pts_str}" fill="{fill_col}" stroke="#111" stroke-width="0.8" opacity="0.85"/>'

            svg_full += '</svg>'
            st.components.v1.html(svg_full, height=650, scrolling=True)
