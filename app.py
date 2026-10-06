import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("📐 Footwear Material Yield Visualizer (Drag & Drop Canvas)")
st.caption("Klik, Geser (Drag), dan Putar Komponen 2 Langsung di Atas Kanvas untuk Membuat Master Interlock Pair")

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
        st.subheader("🛠️ Step 1: Drag & Position Component 2 Directly on Canvas")
        st.caption("Klik objek komponen merah di kanvas untuk menggeser (*drag*) atau memutarnya hingga masuk rapat ke celah komponen biru.")

        col_rot, col_status = st.columns([1, 1])
        with col_rot:
            rot_angle = st.slider("Rotasi Komponen 2 Sebelum Di-drag (°)", 0, 360, 180, step=15)

        # Siapkan poligon komponen 2 dengan rotasi pilihan
        p2_rot = rotate(base_poly, rot_angle, origin='center')
        minx, miny, _, _ = p2_rot.bounds
        p2_zero = translate(p2_rot, xoff=-minx, yoff=-miny)

        # Buat gambar latar belakang kanvas (Komponen 1 Fixed di koordinat awal)
        canvas_w_px = 600
        canvas_h_px = 500
        scale_factor = 15.0 # 1 cm = 15 piksel di kanvas

        bg_img = Image.new("RGBA", (canvas_w_px, canvas_h_px), (245, 247, 250, 255))
        draw = ImageDraw.Draw(bg_img)

        # Gambar Komponen 1 (Fixed)
        pts1 = [(pt[0] * scale_factor + 50, pt[1] * scale_factor + 50) for pt in base_poly.exterior.coords]
        draw.polygon(pts1, fill=(51, 136, 255, 180), outline="black")

        # Inisialisasi posisi awal Komponen 2 di kanvas (dapat di-drag)
        pts2_init = [(pt[0] * scale_factor + 50 + (bw * scale_factor * 0.5), pt[1] * scale_factor + 50 + (bh * scale_factor * 0.5)) for pt in p2_zero.exterior.coords]
        
        initial_drawing = {
            "version": "4.4.0",
            "objects": [
                {
                    "type": "polygon",
                    "version": "4.4.0",
                    "originX": "left",
                    "originY": "top",
                    "left": 50 + (bw * scale_factor * 0.5),
                    "top": 50 + (bh * scale_factor * 0.5),
                    "width": bw * scale_factor,
                    "height": bh * scale_factor,
                    "fill": "rgba(255, 68, 68, 0.7)",
                    "stroke": "black",
                    "strokeWidth": 1,
                    "points": [{"x": pt[0] * scale_factor, "y": pt[1] * scale_factor} for pt in p2_zero.exterior.coords]
                }
            ]
        }

        # TAMPILKAN KANVAS INTERAKTIF (DRAG & DROP)
        canvas_result = st_canvas(
            fill_color="rgba(255, 68, 68, 0.7)",
            stroke_width=1,
            background_image=bg_img,
            initial_drawing=initial_drawing,
            update_streamlit=True,
            height=canvas_h_px,
            width=canvas_w_px,
            drawing_mode="transform", # Mode transform memungkinkan drag, scaling, dan rotasi
            key="interlock_canvas",
        )

        # TANGKAP POSISI HASIL DRAG USER DARI KANVAS
        shift_x_cm = (bw * 0.5)
        shift_y_cm = (bh * 0.5)

        if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
            objs = canvas_result.json_data["objects"]
            if len(objs) > 0:
                dragged_obj = objs[0]
                left_px = dragged_obj.get("left", 50)
                top_px = dragged_obj.get("top", 50)
                
                # Konversi piksel kanvas kembali ke centimeter
                shift_x_cm = (left_px - 50) / scale_factor
                shift_y_cm = (top_px - 50) / scale_factor

        poly2_custom = translate(p2_zero, xoff=shift_x_cm, yoff=shift_y_cm)

        # STATUS DETEKSI TABRAKAN
        has_overlap = base_poly.intersects(poly2_custom)
        with col_status:
            if has_overlap:
                st.error("❌ Status: Posisi bertabrakan (*overlap*)! Geser sedikit lagi.")
            else:
                st.success("✅ Status: Interlock Pasangan Aman & Presisi!")

        # Hitung Bounding Box Pasangan Master
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

        if st.button("🚀 Run Full Sheet Nesting Simulation", type="primary"):
            placed_polygons = []
            total_pattern_area = 0.0

            total_items = target_pairs * 2
            item_idx = 0

            # GENERATE LAYOUT BERDASARKAN HASIL DRAG & DROP USER
            if "ROWs" in nesting_mode:
                curr_y = margin
                while item_idx < total_items and (curr_y + pair_h) <= (sheet_length - margin):
                    curr_x = margin
                    while item_idx < total_items and (curr_x + pair_w) <= (sheet_width - margin):
                        p1 = translate(poly1_unit, xoff=curr_x, yoff=curr_y)
                        placed_polygons.append((p1, 0))
                        total_pattern_area += p1.area
                        item_idx += 1

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

            # METRIK KALKULASI PROCOST
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

            # VISUALISASI HASIL LEMBARAN MATERIAL
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
            plt.title(f"Custom Interactive Layout ({nesting_mode}) | Comp. Yield: {component_yield:.1f}% | Pairs: {pairs_completed}", fontsize=12)
            plt.xlabel("Width (cm)")
            plt.ylabel("Length (cm)")
            
            st.pyplot(fig)
