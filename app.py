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

st.title("⚡ Footwear Material Yield Visualizer (Auto Contour Nesting)")
st.caption("Algoritma Auto-Nesting Geometris -> Menata & Merapatkan Komponen Otomatis Bebas Tabrakan")

# ============================================================
# SIDEBAR PARAMETER
# ============================================================

st.sidebar.header("⚙️ Parameter Lembaran & Nesting")

sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
inter_gap = st.sidebar.number_input("Jarak Antar Pola / Interlacing Gap (cm)", value=0.1, step=0.05)
target_pairs = st.sidebar.number_input("Jumlah Pasang Target (Pairs)", value=50, min_value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Opsi Rotasi Nesting")
allow_flip = st.sidebar.checkbox("Izinkan Rotasi Pasangan (0° & 180° / Interlock)", value=True)

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

# ============================================================
# AUTOMATIC DYNAMIC NESTING ENGINE
# ============================================================

def run_auto_nesting(base_poly, sheet_w, sheet_l, edge_margin, gap, total_pcs, allow_rot_180):
    placed_items = []
    placed_buffers = []
    total_area = 0.0

    # Orientasi yang disiapkan (Normal 0° & Inverted 180°)
    poly_0 = translate(base_poly, xoff=-base_poly.bounds[0], yoff=-base_poly.bounds[1])
    poly_180 = rotate(poly_0, 180, origin='center')
    poly_180 = translate(poly_180, xoff=-poly_180.bounds[0], yoff=-poly_180.bounds[1])

    variants = [poly_0]
    if allow_rot_180:
        variants.append(poly_180)

    step_grid = 0.25  # Resolusi pencarian presisi (cm)

    for i in range(total_pcs):
        best_cand = None
        best_buf = None
        best_y = float('inf')
        best_x = float('inf')

        # Coba tiap orientasi rotasi
        for var_idx, p_var in enumerate(variants):
            w_var = p_var.bounds[2] - p_var.bounds[0]
            h_var = p_var.bounds[3] - p_var.bounds[1]

            # Loop Y dari atas lembaran ke bawah
            curr_y = edge_margin
            while curr_y + h_var <= sheet_l - edge_margin:
                # Loop X dari kiri ke kanan
                curr_x = edge_margin
                while curr_x + w_var <= sheet_w - edge_margin:
                    cand = translate(p_var, xoff=curr_x, yoff=curr_y)
                    cand_buf = cand.buffer(gap / 2)

                    # Check tabrakan dengan komponen yang sudah terpasang
                    has_collision = False
                    for existing_buf in placed_buffers:
                        if cand_buf.intersects(existing_buf):
                            has_collision = True
                            break

                    if not has_collision:
                        # Dapatkan posisi paling rapat (paling atas & paling kiri)
                        if (curr_y < best_y) or (abs(curr_y - best_y) < 0.01 and curr_x < best_x):
                            best_y = curr_y
                            best_x = curr_x
                            best_cand = (cand, var_idx)
                            best_buf = cand_buf

                        # Begitu dapat posisi X aman di Y ini, lanjut ke Y berikutnya untuk efisiensi
                        break

                    curr_x += step_grid

                # Jika sudah menemukan opsi di posisi Y yang sangat rapat, pertimbangkan
                if best_cand is not None and curr_y > best_y + 2.0:
                    break

                curr_y += step_grid

        if best_cand is not None:
            placed_items.append(best_cand)
            placed_buffers.append(best_buf)
            total_area += best_cand[0].area
        else:
            # Tidak ada ruang tersisa di lembaran
            break

    return placed_items, total_area

# ============================================================
# MAIN APP
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

        total_pcs_target = target_pairs * 2

        st.markdown("---")
        st.subheader("🤖 Automatic Nesting & Layout Generation")
        st.info(f"Target Komponen: **{total_pcs_target} pcs** ({target_pairs} Pasang) | Ukuran Pola: **{bw:.2f} x {bh:.2f} cm**")

        if st.button("🚀 Jalankan Auto-Nesting (Rapatkan Komponen)", type="primary"):
            with st.spinner("Sedang menghitung kontur & menata pola secara otomatis..."):
                placed_polygons, total_pattern_area = run_auto_nesting(
                    base_poly,
                    sheet_width,
                    sheet_length,
                    margin,
                    inter_gap,
                    total_pcs_target,
                    allow_flip
                )

            if not placed_polygons:
                st.error("Gagal menata pola. Pastikan ukuran lembaran cukup besar dibanding ukuran komponen.")
            else:
                # SUMMARY METRICS
                total_sheet_area = sheet_width * sheet_length
                max_used_y = max([p.bounds[3] for p, _ in placed_polygons])
                used_sheet_area = sheet_width * max_used_y if max_used_y > 0 else total_sheet_area

                component_yield = (total_pattern_area / used_sheet_area) * 100 if used_sheet_area > 0 else 0.0
                overall_sheet_yield = (total_pattern_area / total_sheet_area) * 100
                total_waste = 100.0 - component_yield

                pairs_completed = len(placed_polygons) // 2
                consumption_per_pair = (used_sheet_area / 10000) / max(pairs_completed, 1)

                st.markdown("### 📊 Summary Yield & Consuption Material")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Komponen Terpasang", f"{len(placed_polygons)} pcs ({pairs_completed} pairs)")
                m2.metric("Total Net Area", f"{total_pattern_area:.1f} cm²")
                m3.metric("Component Yield", f"{component_yield:.2f} %")
                m4.metric("Overall Sheet Yield", f"{overall_sheet_yield:.2f} %")
                m5.metric("Cutting Waste", f"{total_waste:.2f} %")

                st.success(f"💡 **Consumption Rate:** {consumption_per_pair:.4f} m² / pair | Panjang Bahan Terpakai: {max_used_y:.1f} cm dari {sheet_length:.1f} cm")

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
                for poly, variant_idx in placed_polygons:
                    pts = list(poly.exterior.coords)
                    pts_str = " ".join([f"{p[0] * scale_f:.1f},{p[1] * scale_f:.1f}" for p in pts])
                    fill_col = colors[variant_idx % 2]
                    svg_full += f'<polygon points="{pts_str}" fill="{fill_col}" stroke="#111" stroke-width="0.8" opacity="0.85"/>'

                svg_full += '</svg>'
                st.components.v1.html(svg_full, height=650, scrolling=True)
