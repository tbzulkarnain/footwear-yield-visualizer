import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("📐 Footwear Material Yield Visualizer (Custom Interlock Master)")
st.caption("Atur Posisi Interlock Master (2 Pcs) -> Generate Full Sheet Nesting")

# --- SIDEBAR PARAMETER SHEET ---
st.sidebar.header("⚙️ Parameter Lembaran Material")
sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
inter_gap = st.sidebar.number_input("Jarak Antar Pola / Interlacing Gap (cm)", value=0.5, step=0.1)

st.sidebar.header("🧩 Mode Nesting (ProCost Standard)")
nesting_mode = st.sidebar.selectbox(
    "Pilih Mode Duplikasi:",
    [
        "IP - ROWs (Interlock Pair Rows)",
        "IP - COLUMNs (Interlock Pair Columns)",
        "P - ROWs (Parallel Rows)",
        "P - COLUMNs (Parallel Columns)"
    ]
)

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

        st.markdown("---")
        st.subheader("🛠️ Step 1: Adjust Master Interlock Pair (2 Pcs)")
        st.caption("Atur posisi & rotasi komponen ke-2 agar mengunci rapat ke komponen ke-1 tanpa bertabrakan.")

        col_control, col_preview = st.columns([1, 1])

        with col_control:
            rot_angle = st.slider("Rotasi Komponen 2 (Derajat)", 0, 360, 180, step=5)
            offset_x = st.slider("Pergeseran X (cm)", -float(bw), float(bw * 1.5), float(bw * 0.4), step=0.1)
            offset_y = st.slider("Pergeseran Y (cm)", -float(bh), float(bh * 1.5), float(bh * 0.5), step=0.1)

            # Buat Komponen 2 sesuai kontrol user
            poly2_rot = rotate(base_poly, rot_angle, origin='center')
            minx, miny, _, _ = poly2_rot.bounds
            poly2_zero = translate(poly2_rot, xoff=-minx, yoff=-miny)
            poly2_custom = translate(poly2_zero, xoff=offset_x, yoff=offset_y)

            # Cek status overlap/tabrakan
            is_overlapping = base_poly.intersects(poly2_custom)
            if is_overlapping:
                st.warning("⚠️ Peringatan: Posisi komponen ke-2 bertabrakan (overlap) dengan komponen ke-1!")
            else:
                st.success("✅ Posisi Interlock Aman (Tidak bertabrakan).")

        with col_preview:
            # Render visualisasi unit master pasangan
            fig_pair, ax_p = plt.subplots(figsize=(5, 5))
            x1, y1 = base_poly.exterior.xy
            x2, y2 = poly2_custom.exterior.xy

            ax_p.fill(x1, y1, alpha=0.7, fc='#88CCEE', ec='black', label='Pcs 1 (Fixed)')
            ax_p.fill(x2, y2, alpha=0.7, fc='#CC6677', ec='black', label='Pcs 2 (Adjusted)')
            ax_p.set_aspect('equal')
            ax_p.legend(loc='upper right')
            plt.title("Master Interlock Pair Preview")
            st.pyplot(fig_pair)

        # Hitung bounding box unit pasangan master
        p_minx = min(base_poly.bounds[0], poly2_custom.bounds[0])
        p_miny = min(base_poly.bounds[1], poly2_custom.bounds[1])
        p_maxx = max(base_poly.bounds[2], poly2_custom.bounds[2])
        p_maxy = max(base_poly.bounds[3], poly2_custom.bounds[3])

        pair_w = p_maxx - p_minx
        pair_h = p_maxy - p_miny

        poly1_unit = translate(base_poly, xoff=-p_minx, yoff=-p_miny)
        poly2_unit = translate(poly2_custom, xoff=-p_minx, yoff=-p_miny)

        st.markdown("---")
        st.subheader("🚀 Step 2: Generate Full Sheet Nesting")

        if st.button("Generate Full Sheet Layout", type="primary"):
            placed_polygons = []
            total_pattern_area = 0.0

            total_items = target_pairs * 2
            item_idx = 0

            # SIMULASI DUPUKASI FULL SHEET
            if "ROWs" in nesting_mode:
                curr_y = margin
                while item_idx < total_items and (curr_y + pair_h) <= (sheet_length - margin):
                    curr_x = margin
                    while item_idx < total_items and (curr_x + pair_w) <= (sheet_width - margin):
                        # Pcs 1
                        p1 = translate(poly1_unit, xoff=curr_x, yoff=curr_y)
                        placed_polygons.append((p1, 0))
                        total_pattern_area += p1.area
                        item_idx += 1

                        # Pcs 2
                        if item_idx < total_items:
                            p2 = translate(poly2_unit, xoff=curr_x, yoff=curr_y)
                            placed_polygons.append((p2, 1))
                            total_pattern_area += p2.area
                            item_idx += 1

                        curr_x += pair_w + inter_gap
                    curr_y += pair_h + inter_gap
            else: # COLUMNs
                curr_x = margin
                while item_idx < total_items and (curr_x + pair_w) <= (sheet_width - margin):
                    curr_y = margin
                    while item_idx < total_items and (curr_y + pair_h) <= (sheet_length - margin):
                        p1 = translate(poly1_unit, xoff=curr_x, yoff=curr_y)
                        placed_polygons.append((p1, 0))
                        total_pattern_area += p1.area
                        item_idx += 1

                        if item_idx < total_items:
                            p2 = translate(poly2_unit, xoff=curr_x, yoff=curr_y)
                            placed_polygons.append((p2, 1))
                            total_pattern_area += p2.area
                            item_idx += 1

                        curr_y += pair_h + inter_gap
                    curr_x += pair_w + inter_gap

            # METRIK YIELD
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

            st.info(f"💡 **Consumption Rate:** {consumption_per_pair:.4f} m² / pair | Panjang Bahan Terpakai: {max_used_y:.1f} cm")

            # VISUALISASI HASIL FULL SHEET
            fig, ax = plt.subplots(figsize=(14, 8))
            
            sheet_rect = patches.Rectangle((0, 0), sheet_width, sheet_length, linewidth=2, edgecolor='black', facecolor='#F8F9FA')
            ax.add_patch(sheet_rect)
            
            margin_rect = patches.Rectangle((margin, margin), sheet_width - (2*margin), sheet_length - (2*margin), 
                                            linewidth=1, edgecolor='red', linestyle='--')
            ax.add_patch(margin_rect)

            if max_used_y > 0:
                ax.axhline(y=max_used_y, color='blue', linestyle=':', linewidth=1.5, label='Actual Cut Line')

            colors = ['#88CCEE', '#CC6677']

            for poly, idx in placed_polygons:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.85, fc=colors[idx % 2], ec='black', linewidth=1)

            ax.set_xlim(-5, sheet_width + 5)
            ax.set_ylim(-5, sheet_length + 5)
            ax.set_aspect('equal')
            plt.title(f"Custom Master Layout ({nesting_mode}) | Comp. Yield: {component_yield:.1f}% | Pairs: {pairs_completed}", fontsize=12)
            plt.xlabel("Width (cm)")
            plt.ylabel("Length (cm)")
            
            st.pyplot(fig)
