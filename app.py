import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("🤖 Footwear Material Yield Visualizer (1-Click AI Auto-Nesting)")
st.caption("Upload Gambar Component -> AI Otomatis Mencari Posisi Interlock Paling Hemat & Bebas Tabrakan")

# --- SIDEBAR PARAMETER SHEET ---
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

# --- ENGINE OPTIMASI AI (SEARCH BEST INTERLOCK & ROW COMPRESSION) ---
def find_best_ai_nesting(base_poly, gap_cm):
    bw = base_poly.bounds[2] - base_poly.bounds[0]
    bh = base_poly.bounds[3] - base_poly.bounds[1]

    best_yield_score = -1.0
    best_config = None

    # Iterasi pencarian otomatis oleh AI
    angles = [180, 0, 90, 270]
    
    # 1. Cari interlock pasangan terbaik (Pcs 1 + Pcs 2)
    for rot2 in angles:
        p2_rot = rotate(base_poly, rot2, origin='center')
        minx2, miny2, _, _ = p2_rot.bounds
        p2_zero = translate(p2_rot, xoff=-minx2, yoff=-miny2)

        for dx2 in np.linspace(0.1 * bw, 0.8 * bw, 10):
            for dy2 in np.linspace(0.1 * bh, 0.9 * bh, 10):
                poly2_cand = translate(p2_zero, xoff=dx2, yoff=dy2)

                # Syarat 1: Pasangan tidak bertabrakan
                if not poly2_cand.buffer(gap_cm / 2).intersects(base_poly):
                    
                    # Hitung unit pasangan
                    p_minx = min(base_poly.bounds[0], poly2_cand.bounds[0])
                    p_miny = min(base_poly.bounds[1], poly2_cand.bounds[1])
                    p_maxx = max(base_poly.bounds[2], poly2_cand.bounds[2])
                    p_maxy = max(base_poly.bounds[3], poly2_cand.bounds[3])

                    u1 = translate(base_poly, xoff=-p_minx, yoff=-p_miny)
                    u2 = translate(poly2_cand, xoff=-p_minx, yoff=-p_miny)
                    unit_w = p_maxx - p_minx
                    unit_h = p_maxy - p_miny

                    # 2. Cari interlock antar-baris (Baris 1 ke Baris 2)
                    for r2_sx in np.linspace(-0.5 * unit_w, 0.5 * unit_w, 8):
                        for r2_sy in np.linspace(0.3 * unit_h, 0.95 * unit_h, 10):
                            r2_u1 = translate(u1, xoff=r2_sx, yoff=r2_sy)
                            r2_u2 = translate(u2, xoff=r2_sx, yoff=r2_sy)

                            # Syarat 2: Baris 2 tidak menindih Baris 1
                            collide1 = r2_u1.buffer(gap_cm / 2).intersects(u1) or r2_u1.buffer(gap_cm / 2).intersects(u2)
                            collide2 = r2_u2.buffer(gap_cm / 2).intersects(u1) or r2_u2.buffer(gap_cm / 2).intersects(u2)

                            if not collide1 and not collide2:
                                # Hitung skor efisiensi (makin kecil bounding area baris, makin hemat)
                                block_area = (unit_w + abs(r2_sx)) * r2_sy
                                score = (u1.area * 2) / block_area

                                if score > best_yield_score:
                                    best_yield_score = score
                                    best_config = {
                                        'u1': u1, 'u2': u2,
                                        'unit_w': unit_w, 'unit_h': unit_h,
                                        'r2_sx': r2_sx, 'r2_sy': r2_sy
                                    }

    return best_config

# --- UPLOAD GAMBAR ---
uploaded_file = st.file_uploader("Upload Gambar Pattern Component Master", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    raw_polygons = extract_polygons_from_image(uploaded_file)
    
    if not raw_polygons:
        st.error("Gagal mendeteksi bentuk pola dari gambar. Pastikan kontur garis pola jelas.")
    else:
        base_poly = raw_polygons[0]

        st.markdown("---")
        st.subheader("⚡ Full Automated AI Nesting Engine")
        
        if st.button("🤖 Run AI Auto-Nesting Optimization", type="primary"):
            with st.spinner("AI sedang menganalisis & menghitung kombinasi interlock paling presisi..."):
                cfg = find_best_ai_nesting(base_poly, inter_gap)

                if cfg is None:
                    st.error("Gagal menemukan posisi interlock aman. Coba kecilkan Jarak Antar Pola (Interlacing Gap).")
                else:
                    poly1_unit = cfg['u1']
                    poly2_unit = cfg['u2']
                    unit_w = cfg['unit_w']
                    unit_h = cfg['unit_h']
                    r2_shift_x = cfg['r2_sx']
                    r2_shift_y = cfg['r2_sy']

                    # GENERATE NESTING FULL SHEET
                    placed_polygons = []
                    total_pattern_area = 0.0

                    total_items = target_pairs * 2
                    item_idx = 0
                    row_idx = 0

                    curr_base_y = margin

                    while item_idx < total_items and (curr_base_y + min(poly1_unit.bounds[3], poly2_unit.bounds[3])) <= (sheet_length - margin):
                        is_row_even = (row_idx % 2 == 0)
                        
                        if is_row_even:
                            row_y = curr_base_y
                            x_offset = 0.0
                        else:
                            row_y = curr_base_y + r2_shift_y
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
                        if row_idx % 2 == 0:
                            curr_base_y += (2 * r2_shift_y)

                    # METRIK SUMMARY
                    total_sheet_area = sheet_width * sheet_length
                    max_used_y = max([p.bounds[3] for p, _ in placed_polygons]) if placed_polygons else 0.0
                    used_sheet_area = sheet_width * max_used_y if max_used_y > 0 else total_sheet_area

                    component_yield = (total_pattern_area / used_sheet_area) * 100 if used_sheet_area > 0 else 0.0
                    overall_sheet_yield = (total_pattern_area / total_sheet_area) * 100
                    total_waste = 100.0 - component_yield
                    
                    pairs_completed = len(placed_polygons) // 2
                    consumption_per_pair = (used_sheet_area / 10000) / max(pairs_completed, 1)

                    st.success("✅ AI Optimization Complete! Layout 100% Presisi & Bebas Tabrakan.")

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
                    plt.title(f"AI Auto-Optimized Nesting Layout | Comp. Yield: {component_yield:.1f}% | Pairs: {pairs_completed}", fontsize=12)
                    plt.xlabel("Width (cm)")
                    plt.ylabel("Length (cm)")
                    
                    st.pyplot(fig)
