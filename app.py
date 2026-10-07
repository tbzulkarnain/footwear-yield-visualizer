import streamlit as st
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("⚡ Footwear Material Yield Visualizer (Instant Auto-Nesting)")
st.caption("Upload Gambar Pattern Component -> Layout Langsung Tampil Instan")

# --- SIDEBAR PARAMETER SHEET ---
st.sidebar.header("⚙️ Parameter Lembaran Material")
sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
inter_gap = st.sidebar.number_input("Jarak Antar Pola / Interlacing Gap (cm)", value=0.1, step=0.05)

target_pairs = st.sidebar.number_input("Jumlah Pasang Target (Pairs)", value=50, min_value=1, step=1)

# --- FUNGSI OPENCV DENGAN CACHING (BIAR GA RE-RUN BERULANG) ---
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
    except Exception as e:
        return []

# --- ENGINE SMART AUTO-SNAP (INSTAN) ---
def smart_auto_snap(base_poly, gap_cm):
    p2_rot = rotate(base_poly, 180, origin='center')
    minx2, miny2, _, _ = p2_rot.bounds
    p2_zero = translate(p2_rot, xoff=-minx2, yoff=-miny2)

    bw = base_poly.bounds[2] - base_poly.bounds[0]
    bh = base_poly.bounds[3] - base_poly.bounds[1]

    dx2 = bw * 0.35
    dy2 = bh * 0.2
    poly2 = translate(p2_zero, xoff=dx2, yoff=dy2)

    p_minx = min(base_poly.bounds[0], poly2.bounds[0])
    p_miny = min(base_poly.bounds[1], poly2.bounds[1])
    p_maxx = max(base_poly.bounds[2], poly2.bounds[2])
    p_maxy = max(base_poly.bounds[3], poly2.bounds[3])

    u1 = translate(base_poly, xoff=-p_minx, yoff=-p_miny)
    u2 = translate(poly2, xoff=-p_minx, yoff=-p_miny)

    unit_w = p_maxx - p_minx
    unit_h = p_maxy - p_miny

    u1_buf = u1.buffer(gap_cm / 2)
    u2_buf = u2.buffer(gap_cm / 2)

    r2_sx = unit_w * 0.5
    y_min = unit_h * 0.2
    y_max = unit_h * 1.2
    safe_y = y_max

    for test_y in np.linspace(y_min, y_max, 20):
        r2_u1 = translate(u1_buf, xoff=r2_sx, yoff=test_y)
        r2_u2 = translate(u2_buf, xoff=r2_sx, yoff=test_y)

        r3_u1 = translate(u1_buf, xoff=0, yoff=test_y * 2)
        r3_u2 = translate(u2_buf, xoff=0, yoff=test_y * 2)

        collide_12 = r2_u1.intersects(u1_buf) or r2_u1.intersects(u2_buf) or r2_u2.intersects(u1_buf) or r2_u2.intersects(u2_buf)
        collide_23 = r3_u1.intersects(r2_u1) or r3_u1.intersects(r2_u2) or r3_u2.intersects(r2_u1) or r3_u2.intersects(r2_u2)

        if not collide_12 and not collide_23:
            safe_y = test_y
            break

    return {
        'u1': u1, 'u2': u2,
        'unit_w': unit_w, 'unit_h': unit_h,
        'r2_sx': r2_sx, 'r2_sy': safe_y
    }

# --- RENDERER SVG SUPER FAST ---
def generate_svg_layout(sheet_w, sheet_l, margin_v, placed_polys, max_y):
    scale = 8
    svg_w = sheet_w * scale
    svg_h = sheet_l * scale

    svg_code = f'<svg width="100%" height="auto" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background-color: #F8F9FA; border: 2px solid black;">'
    
    m_x = margin_v * scale
    m_y = margin_v * scale
    m_w = (sheet_w - 2 * margin_v) * scale
    m_h = (sheet_l - 2 * margin_v) * scale
    svg_code += f'<rect x="{m_x}" y="{m_y}" width="{m_w}" height="{m_h}" fill="none" stroke="red" stroke-dasharray="4" stroke-width="1.5"/>'

    if max_y > 0:
        c_y = max_y * scale
        svg_code += f'<line x1="0" y1="{c_y}" x2="{svg_w}" y2="{c_y}" stroke="blue" stroke-dasharray="2" stroke-width="2"/>'

    colors = ['#3388ff', '#ff4444']
    for poly, idx in placed_polys:
        pts = list(poly.exterior.coords)
        pts_str = " ".join([f"{p[0]*scale},{p[1]*scale}" for p in pts])
        fill_col = colors[idx % 2]
        svg_code += f'<polygon points="{pts_str}" fill="{fill_col}" stroke="black" stroke-width="1" opacity="0.85"/>'

    svg_code += '</svg>'
    return svg_code

# --- UPLOAD GAMBAR & AUTO EXECUTE ---
uploaded_file = st.file_uploader("Upload Gambar Pattern Component Master", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    raw_polygons = extract_polygons_from_bytes(file_bytes)
    
    if not raw_polygons:
        st.error("Gagal mendeteksi bentuk pola dari gambar. Pastikan gambar kontur pola kontras dan jelas.")
    else:
        base_poly = raw_polygons[0]

        # PROSES OTOMATIS TANPA PILIH TOMBOL
        cfg = smart_auto_snap(base_poly, inter_gap)

        poly1_unit = cfg['u1']
        poly2_unit = cfg['u2']
        unit_w = cfg['unit_w']
        unit_h = cfg['unit_h']
        r2_shift_x = cfg['r2_sx']
        r2_shift_y = cfg['r2_sy']

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

        total_sheet_area = sheet_width * sheet_length
        max_used_y = max([p.bounds[3] for p, _ in placed_polygons]) if placed_polygons else 0.0
        used_sheet_area = sheet_width * max_used_y if max_used_y > 0 else total_sheet_area

        component_yield = (total_pattern_area / used_sheet_area) * 100 if used_sheet_area > 0 else 0.0
        overall_sheet_yield = (total_pattern_area / total_sheet_area) * 100
        total_waste = 100.0 - component_yield
        
        pairs_completed = len(placed_polygons) // 2
        consumption_per_pair = (used_sheet_area / 10000) / max(pairs_completed, 1)

        st.success("✅ Auto-Nesting Selesai!")

        st.markdown("### 📊 Yield & Material Consumption Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Komponen Terpasang", f"{len(placed_polygons)} pcs ({pairs_completed} pairs)")
        m2.metric("Total Net Area", f"{total_pattern_area:.1f} cm²")
        m3.metric("Component Yield", f"{component_yield:.2f} %")
        m4.metric("Overall Sheet Yield", f"{overall_sheet_yield:.2f} %")
        m5.metric("Cutting Waste", f"{total_waste:.2f} %")

        st.info(f"💡 **Consumption Rate:** {consumption_per_pair:.4f} m² / pair | Panjang Bahan Terpakai: {max_used_y:.1f} cm dari {sheet_length:.1f} cm")

        # VISUALISASI SVG INSTAN
        svg_html = generate_svg_layout(sheet_width, sheet_length, margin, placed_polygons, max_used_y)
        st.components.v1.html(svg_html, height=650, scrolling=True)
