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

st.title("⚡ Footwear Material Yield Visualizer (Exact Grid Repeat)")
st.caption("3-Component Master Setup → Repeat Master Unit Ke Lembaran Utuh")

# ============================================================
# SIDEBAR PARAMETER
# ============================================================

st.sidebar.header("⚙️ Parameter Lembaran Material")

sheet_width = st.sidebar.number_input(
    "Lebar Material / Sheet Width (cm)",
    value=140.0,
    step=5.0
)

sheet_length = st.sidebar.number_input(
    "Panjang Material / Sheet Length (cm)",
    value=100.0,
    step=5.0
)

margin = st.sidebar.number_input(
    "Margin Pinggir / Edge Gap (cm)",
    value=1.0,
    step=0.5
)

inter_gap = st.sidebar.number_input(
    "Jarak Antar Pola / Interlacing Gap (cm)",
    value=0.1,
    step=0.05
)

target_pairs = st.sidebar.number_input(
    "Jumlah Pasang Target (Pairs)",
    value=50,
    min_value=1,
    step=1
)


# ============================================================
# EXTRACT POLYGONS
# ============================================================

@st.cache_data
def extract_polygons_from_bytes(file_bytes, dpi=96):
    try:
        img = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_UNCHANGED
        )

        if img is None:
            return []

        if len(img.shape) == 3 and img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_svg_3pcs_preview(p1, p2, p3, width_cm=50, height_cm=40):
    scale = 10
    svg_w = width_cm * scale
    svg_h = height_cm * scale

    svg_code = f'''
    <svg width="100%" height="auto" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#F8F9FA; border:2px dashed #666; border-radius:8px;">
    '''

    items = [(p1, "#3388ff"), (p2, "#ff4444"), (p3, "#28a745")]

    for poly, col in items:
        pts = list(poly.exterior.coords)
        pts_str = " ".join([f"{p[0] * scale:.1f},{p[1] * scale:.1f}" for p in pts])
        svg_code += f'<polygon points="{pts_str}" fill="{col}" stroke="#111" stroke-width="1" opacity="0.85"/>'

    svg_code += "</svg>"
    return svg_code


def polygons_collide(poly_a, poly_b, gap=0.1):
    safe_a = poly_a.buffer(gap / 2)
    safe_b = poly_b.buffer(gap / 2)
    return safe_a.intersects(safe_b)


# ============================================================
# MAIN APP LOGIC
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Gambar Pattern Component Master",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    raw_polygons = extract_polygons_from_bytes(file_bytes)

    if not raw_polygons:
        st.error("Gagal mendeteksi bentuk pola dari gambar. Pastikan garis kontur pola jelas.")
    else:
        base_poly = raw_polygons[0]
        bw = base_poly.bounds[2] - base_poly.bounds[0]
        bh = base_poly.bounds[3] - base_poly.bounds[1]

        # ====================================================
        # STEP 1: OPERATOR ADJUSTMENT (3 COMPONENTS)
        # ====================================================

        st.markdown("---")
        st.subheader("🛠️ Step 1: Atur Posisi Master 3 Komponen")

        col_ctrl, col_prev = st.columns([1.1, 0.9])

        with col_ctrl:
            st.markdown("##### 🔵 Komponen 1 (Baris 1 - Pcs 1)")
            c1_1, c1_2, c1_3 = st.columns(3)
            with c1_1:
                rot1 = st.slider("Rotasi Pcs 1 (°)", 0, 360, 90, step=5)
            with c1_2:
                shift_x1 = st.slider("Geser X Pcs 1", -float(bw * 2), float(bw * 2), 0.0, step=0.1)
            with c1_3:
                shift_y1 = st.slider("Geser Y Pcs 1", -float(bh * 2), float(bh * 2), 0.0, step=0.1)

            st.markdown("##### 🔴 Komponen 2 (Baris 1 - Pcs 2)")
            c2_1, c2_2, c2_3 = st.columns(3)
            with c2_1:
                rot2 = st.slider("Rotasi Pcs 2 (°)", 0, 360, 270, step=5)
            with c2_2:
                shift_x2 = st.slider("Geser X Pcs 2", -float(bw * 2), float(bw * 2), 4.5, step=0.1)
            with c2_3:
                shift_y2 = st.slider("Geser Y Pcs 2", -float(bh * 2), float(bh * 2), -8.5, step=0.1)

            st.markdown("##### 🟢 Komponen 3 (Awal Baris 2)")
            c3_1, c3_2, c3_3 = st.columns(3)
            with c3_1:
                rot3 = st.slider("Rotasi Pcs 3 (°)", 0, 360, rot1, step=5)
            with c3_2:
                r2_shift_x = st.slider("Geser X Pcs 3", -float(bw * 2), float(bw * 2), 8.1, step=0.1)
            with c3_3:
                r2_shift_y = st.slider("Geser Y Pcs 3", -float(bh * 2), float(bh * 2), 6.1, step=0.1)

        # BENTUK KOMPONEN BERDASARKAN PARAMETER SLIDER
        p1_rot = rotate(base_poly, rot1, origin="center")
        p1_poly = translate(p1_rot, xoff=shift_x1, yoff=shift_y1)

        p2_rot = rotate(base_poly, rot2, origin="center")
        p2_poly = translate(p2_rot, xoff=shift_x2, yoff=shift_y2)

        p3_rot = rotate(base_poly, rot3, origin="center")
        p3_poly = translate(p3_rot, xoff=r2_shift_x, yoff=r2_shift_y)

        # PREVIEW NORMALIZATION
        min_canvas_x = min(p1_poly.bounds[0], p2_poly.bounds[0], p3_poly.bounds[0])
        min_canvas_y = min(p1_poly.bounds[1], p2_poly.bounds[1], p3_poly.bounds[1])

        pad = 5.0
        p1_preview = translate(p1_poly, xoff=-min_canvas_x + pad, yoff=-min_canvas_y + pad)
        p2_preview = translate(p2_poly, xoff=-min_canvas_x + pad, yoff=-min_canvas_y + pad)
        p3_preview = translate(p3_poly, xoff=-min_canvas_x + pad, yoff=-min_canvas_y + pad)

        # DETEKSI TABRAKAN PREVIEW
        collide_12 = polygons_collide(p1_preview, p2_preview, inter_gap)
        collide_13 = polygons_collide(p1_preview, p3_preview, inter_gap)
        collide_23 = polygons_collide(p2_preview, p3_preview, inter_gap)

        with col_prev:
            st.markdown("### 👁️ Preview Master (3 Komponen)")
            if collide_12 or collide_13 or collide_23:
                st.error("⚠️ Terdapat Komponen yang Bertabrakan! Adjust slider sampai posisi aman.")
            else:
                st.success("✅ 3 Komponen Bebas Tabrakan (Layout Safe)")

            pw = max(p1_preview.bounds[2], p2_preview.bounds[2], p3_preview.bounds[2]) + pad
            ph = max(p1_preview.bounds[3], p2_preview.bounds[3], p3_preview.bounds[3]) + pad

            svg_3pcs = generate_svg_3pcs_preview(
                p1_preview,
                p2_preview,
                p3_preview,
                width_cm=max(pw, 25),
                height_cm=max(ph, 25)
            )
            st.components.v1.html(svg_3pcs, height=380, scrolling=False)

        # ====================================================
        # STEP 2: DUPLIKASI REPEAT MASTER UNIT KHUSUS
        # ====================================================

        st.markdown("---")
        st.subheader("🚀 Step 2: Duplikasi Ke Lembaran Utuh")

        if st.button("📊 Repeat Master Setup To Sheet", type="primary"):

            # 1. BENTUK MASTER UNIT SAMA PERSIS SEPERTI STEP 1
            all_master = [p1_poly, p2_poly, p3_poly]

            # Zeroing Master Unit secara utuh
            min_x_m = min(p.bounds[0] for p in all_master)
            min_y_m = min(p.bounds[1] for p in all_master)

            master_p1 = translate(p1_poly, xoff=-min_x_m, yoff=-min_y_m)
            master_p2 = translate(p2_poly, xoff=-min_x_m, yoff=-min_y_m)
            master_p3 = translate(p3_poly, xoff=-min_x_m, yoff=-min_y_m)

            # Hitung Vektor Offset P3 terhadap P1 sebagai arah berulang
            step_repeat_x = master_p3.bounds[0] - master_p1.bounds[0]
            step_repeat_y = master_p3.bounds[1] - master_p1.bounds[1]

            # Lebar Efektif Pasangan P1-P2 untuk pengulangan sejajar X
            pair_width = max(master_p1.bounds[2], master_p2.bounds[2]) - min(master_p1.bounds[0], master_p2.bounds[0])
            step_x = pair_width + inter_gap

            placed_polygons = []
            total_pattern_area = 0.0

            total_items = target_pairs * 2
            item_idx = 0
            row_idx = 0

            # 2. DUPLIKASI GRID DENGAN MENGHORMATI RELASI P1-P2-P3
            while item_idx < total_items:
                is_row_even = (row_idx % 2 == 1)

                row_y = margin + (row_idx * step_repeat_y)
                x_shift = step_repeat_x if is_row_even else 0.0

                if row_y + min(master_p1.bounds[3], master_p2.bounds[3]) > (sheet_length - margin):
                    break

                curr_x = margin + x_shift

                while curr_x < margin:
                    curr_x += step_x

                while item_idx < total_items and (curr_x + pair_width) <= (sheet_width - margin):
                    # Geser P1 & P2 bersamaan sesuai Offset Step 1
                    cand_p1 = translate(master_p1, xoff=curr_x, yoff=row_y)
                    cand_p2 = translate(master_p2, xoff=curr_x, yoff=row_y)

                    p1_valid = (cand_p1.bounds[2] <= sheet_width - margin) and (cand_p1.bounds[3] <= sheet_length - margin) and (cand_p1.bounds[0] >= margin)
                    p2_valid = (cand_p2.bounds[2] <= sheet_width - margin) and (cand_p2.bounds[3] <= sheet_length - margin) and (cand_p2.bounds[0] >= margin)

                    if p1_valid:
                        placed_polygons.append((cand_p1, 0))
                        total_pattern_area += cand_p1.area
                        item_idx += 1

                    if item_idx < total_items and p2_valid:
                        placed_polygons.append((cand_p2, 1))
                        total_pattern_area += cand_p2.area
                        item_idx += 1

                    curr_x += step_x

                row_idx += 1

            # SUMMARY & METRIK
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
