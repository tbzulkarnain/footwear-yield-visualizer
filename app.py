import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

st.set_page_config(page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide")

st.title("⚡ Footwear Material Yield Visualizer (Ultra-Fast Plotly)")
st.caption("Atur Pasangan & Baris secara Interaktif -> Hasil Render Langsung Muncul Instan")

# --- SIDEBAR PARAMETER SHEET ---
st.sidebar.header("⚙️ Parameter Lembaran Material")
sheet_width = st.sidebar.number_input("Lebar Material / Sheet Width (cm)", value=140.0, step=5.0)
sheet_length = st.sidebar.number_input("Panjang Material / Sheet Length (cm)", value=100.0, step=5.0)
margin = st.sidebar.number_input("Margin Pinggir / Edge Gap (cm)", value=1.0, step=0.5)
inter_gap = st.sidebar.number_input("Jarak Antar Pola / Interlacing Gap (cm)", value=0.1, step=0.05)

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

        # ==========================================
        # STEP 1: SET PASANGAN MASTER
        # ==========================================
        st.markdown("---")
        st.subheader("🛠️ Step 1: Atur Pasangan Master (Unit Pair)")

        col_c1, col_c2, col_prev = st.columns([1, 1, 1.2])

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
            shift_y2 = st.slider("Geser Y Pcs 2 (cm)", -float(bh), float(bh * 1.5), float(bh * 0.5), step=0.1, key="sy2")

            p2_rot = rotate(base_poly, rot2, origin='center')
            minx2, miny2, _, _ = p2_rot.bounds
            p2_zero = translate(p2_rot, xoff=-minx2, yoff=-miny2)
            poly2_custom = translate(p2_zero, xoff=shift_x2, yoff=shift_y2)

        # Normalisasi Unit Pair
        p_minx = min(poly1_custom.bounds[0], poly2_custom.bounds[0])
        p_miny = min(poly1_custom.bounds[1], poly2_custom.bounds[1])
        p_maxx = max(poly1_custom.bounds[2], poly2_custom.bounds[2])
        p_maxy = max(poly1_custom.bounds[3], poly2_custom.bounds[3])

        unit_w = p_maxx - p_minx
        unit_h = p_maxy - p_miny

        poly1_unit = translate(poly1_custom, xoff=-p_minx, yoff=-p_miny)
        poly2_unit = translate(poly2_custom, xoff=-p_minx, yoff=-p_miny)

        # PREVIEW PASANGAN MASTER (FAST PLOTLY)
        with col_prev:
            fig_p = go.Figure()

            x1, y1 = poly1_unit.exterior.xy
            fig_p.add_trace(go.Scatter(x=list(x1), y=list(y1), fill="toself", name="Pcs 1 (Biru)", fillcolor="rgba(51, 136, 255, 0.85)", line=dict(color="black", width=1)))

            x2, y2 = poly2_unit.exterior.xy
            fig_p.add_trace(go.Scatter(x=list(x2), y=list(y2), fill="toself", name="Pcs 2 (Merah)", fillcolor="rgba(255, 68, 68, 0.85)", line=dict(color="black", width=1)))

            fig_p.update_layout(title="Preview Master Pair Unit", yaxis=dict(scaleanchor="x", scaleratio=1), height=280, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_p, use_container_width=True)

        # ==========================================
        # STEP 2: SET INTERLOCK BARIS
        # ==========================================
        st.markdown("---")
        st.subheader("🚀 Step 2: Atur Interlock Antar-Baris")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            r2_shift_x = st.slider("↔️ Pergeseran Horizontal Baris 2 (cm)", -float(unit_w), float(unit_w), float(unit_w * 0.25), step=0.1)
        with col_r2:
            r2_shift_y = st.slider("↕️ Jarak Vertikal Antar-Baris (cm)", float(unit_h * 0.3), float(unit_h * 1.2), float(unit_h * 0.7), step=0.1)

        # CEK TABRAKAN BARIS 1 & 2
        r2_p1 = translate(poly1_unit, xoff=r2_shift_x, yoff=r2_shift_y)
        r2_p2 = translate(poly2_unit, xoff=r2_shift_x, yoff=r2_shift_y)

        has_collision = (
            r2_p1.intersects(poly1_unit) or r2_p1.intersects(poly2_unit) or
            r2_p2.intersects(poly1_unit) or r2_p2.intersects(poly2_unit)
        )

        if has_collision:
            st.error("⚠️ Posisi Baris 2 bertabrakan dengan Baris 1! Naikkkan nilai 'Jarak Vertikal Antar-Baris'.")
        else:
            st.success("✅ Layout 100% Bebas Tabrakan!")

        # GENERATE NESTING GEOMETRI
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

        # METRIK SUMMARY
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

        # ==========================================
        # VISUALISASI FULL SHEET LAYOUT (BATCH TRACE RENDERING)
        # ==========================================
        fig_sheet = go.Figure()

        # 1. Batas Lembaran Material
        fig_sheet.add_trace(go.Scatter(
            x=[0, sheet_width, sheet_width, 0, 0],
            y=[0, 0, sheet_length, sheet_length, 0],
            mode="lines",
            name="Sheet",
            line=dict(color="black", width=2)
        ))

        # 2. Batas Margin
        fig_sheet.add_trace(go.Scatter(
            x=[margin, sheet_width - margin, sheet_width - margin, margin, margin],
            y=[margin, margin, sheet_length - margin, sheet_length - margin, margin],
            mode="lines",
            name="Margin",
            line=dict(color="red", width=1, dash="dash")
        ))

        # 3. Batch Combined Arrays (Instan 1-Trace per Warna)
        x_blue, y_blue = [], []
        x_red, y_red = [], []

        for poly, idx in placed_polygons:
            px, py = poly.exterior.xy
            if idx % 2 == 0:
                x_blue.extend(list(px) + [None])
                y_blue.extend(list(py) + [None])
            else:
                x_red.extend(list(px) + [None])
                y_red.extend(list(py) + [None])

        # Render Komponen Biru
        fig_sheet.add_trace(go.Scatter(
            x=x_blue, y=y_blue,
            fill="toself",
            fillcolor="rgba(51, 136, 255, 0.85)",
            line=dict(color="black", width=0.8),
            name="Komponen 1"
        ))

        # Render Komponen Merah
        fig_sheet.add_trace(go.Scatter(
            x=x_red, y=y_red,
            fill="toself",
            fillcolor="rgba(255, 68, 68, 0.85)",
            line=dict(color="black", width=0.8),
            name="Komponen 2"
        ))

        fig_sheet.update_layout(
            title=f"Full Sheet Layout (Instant Batch Render) | Comp. Yield: {component_yield:.1f}% | Pairs: {pairs_completed}",
            xaxis=dict(range=[-5, sheet_width + 5], title="Width (cm)"),
            yaxis=dict(range=[-5, sheet_length + 5], title="Length (cm)", scaleanchor="x", scaleratio=1),
            height=600,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig_sheet, use_container_width=True)
