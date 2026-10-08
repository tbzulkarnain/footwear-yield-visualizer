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

st.title("⚡ Footwear Material Yield Visualizer")
st.markdown("---")

# ============================================================
# TAHAP 1: SETUP PARAMETER UTAMA DI HALAMAN UTAMA (COMPACT GRID)
# ============================================================

st.subheader("📋 Setup Parameter Bahan & Target")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)

with col_p1:
    sheet_width = st.number_input("Lebar Material (cm)", value=140.0, step=5.0)
with col_p2:
    sheet_length = st.number_input("Panjang Material (cm)", value=100.0, step=5.0)
with col_p3:
    margin = st.number_input("Margin Pinggir (cm)", value=1.0, step=0.5)
with col_p4:
    target_pieces = st.number_input("Target Komponen (Pcs)", value=100, min_value=1, step=1)

uploaded_file = st.file_uploader("Upload Gambar Pattern Component Master", type=["png", "jpg", "jpeg"])

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


def generate_svg_preview_grid(items_with_color, width_cm=45, height_cm=30):
    # Skala diset kecil (1.2) agar gambar preview pas dan tidak kepotong
    scale = 1.2
    svg_w = width_cm * scale
    svg_h = height_cm * scale

    svg_code = f'<svg width="100%" height="auto" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#F8F9FA; border:2px dashed #666; border-radius:8px;">'
    
    color_map = {0: '#3388ff', 1: '#ff4444'} # 0 = Biru, 1 = Merah
    for poly, color_idx in items_with_color:
        pts = list(poly.exterior.coords)
        pts_str = " ".join([f"{p[0] * scale:.1f},{p[1] * scale:.1f}" for p in pts])
        col = color_map.get(color_idx % 2, '#3388ff')
        svg_code += f'<polygon points="{pts_str}" fill="{col}" stroke="#111" stroke-width="0.8" opacity="0.85"/>'

    svg_code += "</svg>"
    return svg_code

# ============================================================
# MAIN APP LOGIC (JIKA FILE DI-UPLOAD)
# ============================================================

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
        st.subheader("🛠️ Konfigurasi Tata Letak & Fine-tune Presisi")

        # Layout 2 Kolom Utama (Kontrol di Kiri, Preview di Kanan)
        col_ctrl, col_prev = st.columns([1.1, 0.9])

        with col_ctrl:
            category = st.selectbox(
                "Pilih Kategori ProCost",
                [
                    "Category 1: One Way Straight (1 Arah Lurus)",
                    "Category 2: Two Way Interlock (2 Arah 180°)",
                    "Category 3: One Way Staggered (1 Arah Zig-Zag Baris)",
                    "Category 4: Two Way Staggered (2 Arah Zig-Zag Baris)"
                ]
            )

            is_twoway_cat = "Two Way" in category
            is_staggered_cat = "Staggered" in category

            c_rot1, c_rot2 = st.columns(2)
            with c_rot1:
                rot_p1 = st.number_input("Rotasi Pcs 1 (°)", min_value=0, max_value=360, value=90, step=5)
            with c_rot2:
                if is_twoway_cat:
                    default_rot2 = (rot_p1 + 180) % 360
                    rot_p2 = st.number_input("Rotasi Pcs 2 (°)", min_value=0, max_value=360, value=int(default_rot2), step=5)
                else:
                    rot_p2 = rot_p1

            st.markdown("##### 🎛️ Penyesuaian Spasi Grid (Number Input Presisi)")
            c_ft1, c_ft2 = st.columns(2)
            with c_ft1:
                fine_tune_x = st.number_input("Jarak Kolom / Step X", value=0.0, step=0.1, format="%.2f")
            with c_ft2:
                fine_tune_y = st.number_input("Jarak Baris / Pitch Y", value=0.0, step=0.1, format="%.2f")

            if is_staggered_cat:
                preview_shift_x = st.number_input("Geser Baris 2 (Offset X)", value=0.0, step=0.1, format="%.2f")
            else:
                preview_shift_x = 0.0

        # HITUNG GEOMETRI P1 & P2
        p1 = rotate(base_poly, rot_p1, origin='center')
        p1 = translate(p1, xoff=-p1.bounds[0], yoff=-p1.bounds[1])

        p2 = rotate(base_poly, rot_p2, origin='center')
        p2 = translate(p2, xoff=-p2.bounds[0], yoff=-p2.bounds[1])

        p1_w = p1.bounds[2] - p1.bounds[0]
        p1_h = p1.bounds[3] - p1.bounds[1]
        p2_w = p2.bounds[2] - p2.bounds[0]
        p2_h = p2.bounds[3] - p2.bounds[1]

        step_x_base = p1_w if not is_twoway_cat else max(p1_w, p2_w)
        step_x = step_x_base + fine_tune_x
        
        unit_h = p1_h if not is_twoway_cat else max(p1_h, p2_h)
        pitch_y = unit_h + fine_tune_y

        # BENTUK PREVIEW GRID 2 BARIS
        preview_items = []

        if "Category 1" in category:
            preview_items.append((p1, 0))
            preview_items.append((translate(p1, xoff=step_x, yoff=0), 1))
            preview_items.append((translate(p1, xoff=0, yoff=pitch_y), 0))
            preview_items.append((translate(p1, xoff=step_x, yoff=pitch_y), 1))

        elif "Category 2" in category:
            preview_items.append((p1, 0))
            preview_items.append((translate(p2, xoff=step_x, yoff=0), 1))
            preview_items.append((translate(p1, xoff=0, yoff=pitch_y), 0))
            preview_items.append((translate(p2, xoff=step_x, yoff=pitch_y), 1))

        elif "Category 3" in category:
            default_stagger_x = step_x / 2
            preview_items.append((p1, 0))
            preview_items.append((translate(p1, xoff=step_x, yoff=0), 1))
            row2_y = pitch_y
            row2_x1 = default_stagger_x + preview_shift_x
            row2_x2 = row2_x1 + step_x
            preview_items.append((translate(p1, xoff=row2_x1, yoff=row2_y), 0))
            preview_items.append((translate(p1, xoff=row2_x2, yoff=row2_y), 1))

        else: # Category 4
            default_stagger_x = step_x / 2
            preview_items.append((p1, 0))
            preview_items.append((translate(p2, xoff=step_x, yoff=0), 1))
            row2_y = pitch_y
            row2_x1 = default_stagger_x + preview_shift_x
            row2_x2 = row2_x1 + step_x
            preview_items.append((translate(p1, xoff=row2_x1, yoff=row2_y), 0))
            preview_items.append((translate(p2, xoff=row2_x2, yoff=row2_y), 1))

        # LIVE PREVIEW MASTER (2 BARIS) DI SISI KANAN
        with col_prev:
            st.markdown("##### 👁️ Live Preview Grid (2 Baris)")
            st.info(f"💡 Mode: **{category.split(':')[0]}** | Step X: {step_x:.1f} cm | Pitch Y: {pitch_y:.1f} cm")
            svg_preview = generate_svg_preview_grid(preview_items, width_cm=45, height_cm=30)
            st.components.v1.html(svg_preview, height=220, scrolling=False)

        # ============================================================
        # TAHAP 2: RENDER HASIL PENUH KE LEMBARAN BAHAN
        # ============================================================

        st.markdown("---")
        st.subheader("🚀 Render Hasil Penuh ke Lembaran Bahan")

        if st.button("📊 Proses Render Layout ProCost", type="primary", use_container_width=True):
            placed_polygons = []
            total_pattern_area = 0.0
            total_items = target_pieces
            item_idx = 0
            row_idx = 0

            if "Category 1" in category:
                while item_idx < total_items:
                    row_y = margin + (row_idx * pitch_y)
                    if row_y + p1_h > (sheet_length - margin):
                        break
                    curr_x = margin
                    col_idx = 0
                    while item_idx < total_items and (curr_x + p1_w) <= (sheet_width - margin):
                        cand = translate(p1, xoff=curr_x, yoff=row_y)
                        placed_polygons.append((cand, col_idx % 2))
                        total_pattern_area += cand.area
                        item_idx += 1
                        curr_x += step_x
                        col_idx += 1
                    row_idx += 1

            elif "Category 2" in category:
                h_unit = max(p1_h, p2_h)
                while item_idx < total_items:
                    row_y = margin + (row_idx * pitch_y)
                    if row_y + h_unit > (sheet_length - margin):
                        break
                    curr_x = margin
                    col_idx = 0
                    while item_idx < total_items and curr_x <= (sheet_width - margin):
                        p_curr = p1 if col_idx % 2 == 0 else p2
                        current_w = p1_w if col_idx % 2 == 0 else p2_w
                        cand = translate(p_curr, xoff=curr_x, yoff=row_y)
                        
                        if curr_x >= margin and (curr_x + current_w) <= (sheet_width - margin):
                            placed_polygons.append((cand, col_idx % 2))
                            total_pattern_area += cand.area
                            item_idx += 1
                        curr_x += step_x
                        col_idx += 1
                    row_idx += 1

            elif "Category 3" in category:
                stagger_x = (step_x / 2) + preview_shift_x
                while item_idx < total_items:
                    is_row_even = (row_idx % 2 == 1)
                    row_y = margin + (row_idx * pitch_y)
                    if row_y + p1_h > (sheet_length - margin):
                        break
                    row_start_x = margin + (stagger_x if is_row_even else 0.0)
                    while row_start_x - step_x >= margin:
                        row_start_x -= step_x
                    curr_x = row_start_x
                    col_idx = 0
                    while item_idx < total_items and curr_x <= (sheet_width - margin):
                        if curr_x >= margin and (curr_x + p1_w) <= (sheet_width - margin):
                            cand = translate(p1, xoff=curr_x, yoff=row_y)
                            placed_polygons.append((cand, col_idx % 2))
                            total_pattern_area += cand.area
                            item_idx += 1
                        curr_x += step_x
                        col_idx += 1
                    row_idx += 1

            else: # Category 4
                h_unit = max(p1_h, p2_h)
                stagger_x = (step_x / 2) + preview_shift_x
                while item_idx < total_items:
                    is_row_even = (row_idx % 2 == 1)
                    row_y = margin + (row_idx * pitch_y)
                    if row_y + h_unit > (sheet_length - margin):
                        break
                    row_start_x = margin + (stagger_x if is_row_even else 0.0)
                    while row_start_x - step_x >= margin:
                        row_start_x -= step_x
                    curr_x = row_start_x
                    col_idx = 0
                    while item_idx < total_items and curr_x <= (sheet_width - margin):
                        p_curr = p1 if col_idx % 2 == 0 else p2
                        current_w = p1_w if col_idx % 2 == 0 else p2_w
                        cand = translate(p_curr, xoff=curr_x, yoff=row_y)
                        
                        if curr_x >= margin and (curr_x + current_w) <= (sheet_width - margin):
                            placed_polygons.append((cand, col_idx % 2))
                            total_pattern_area += cand.area
                            item_idx += 1
                        curr_x += step_x
                        col_idx += 1
                    row_idx += 1

            # SUMMARY METRICS
            total_sheet_area = sheet_width * sheet_length
            max_used_y = max([p.bounds[3] for p, _ in placed_polygons]) if placed_polygons else 0.0
            used_sheet_area = sheet_width * max_used_y if max_used_y > 0 else total_sheet_area

            component_yield = (total_pattern_area / used_sheet_area) * 100 if used_sheet_area > 0 else 0.0
            overall_sheet_yield = (total_pattern_area / total_sheet_area) * 100
            total_waste = 100.0 - component_yield

            pieces_completed = len(placed_polygons)
            consumption_per_pcs = (used_sheet_area / 10000) / max(pieces_completed, 1)

            st.markdown("### 📊 Yield & Material Consumption Summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Komponen Terpasang", f"{pieces_completed} pcs")
            m2.metric("Total Net Area", f"{total_pattern_area:.1f} cm²")
            m3.metric("Component Yield", f"{component_yield:.2f} %")
            m4.metric("Overall Sheet Yield", f"{overall_sheet_yield:.2f} %")
            m5.metric("Cutting Waste", f"{total_waste:.2f} %")

            st.info(f"💡 **Consumption Rate:** {consumption_per_pcs:.4f} m² / pcs | Panjang Bahan Terpakai: {max_used_y:.1f} cm dari {sheet_length:.1f} cm")

            # RENDER SVG FULL SHEET (Skala diset ke 1.2 agar seluruh lembaran bahan tampil kecil utuh dan tidak terpotong di layar)
            scale_f = 1.2
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

            for poly, idx in placed_polygons:
                pts = list(poly.exterior.coords)
                pts_str = " ".join([f"{p[0] * scale_f:.1f},{p[1] * scale_f:.1f}" for p in pts])
                fill_col = '#3388ff' if idx % 2 == 0 else '#ff4444'
                svg_full += f'<polygon points="{pts_str}" fill="{fill_col}" stroke="#111" stroke-width="0.8" opacity="0.85"/>'

            svg_full += '</svg>'
            st.components.v1.html(svg_full, height=400, scrolling=True)
