import streamlit as st
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Footwear Material Yield Visualizer",
    page_icon="📐",
    layout="wide"
)

st.title("⚡ Footwear Material Yield Visualizer (Exact Grid Repeat)")
st.caption("3-Component Preview → Duplikasi Matriks Pasangan Utuh Presisi")


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

        # -----------------------------
        # Convert to grayscale
        # -----------------------------

        if len(img.shape) == 3 and img.shape[2] == 4:

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2GRAY
            )

        elif len(img.shape) == 3:

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

        else:

            gray = img

        # -----------------------------
        # Blur
        # -----------------------------

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # -----------------------------
        # Adaptive threshold
        # -----------------------------

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        # -----------------------------
        # Find contours
        # -----------------------------

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        pixels_per_cm = dpi / 2.54

        extracted_polygons = []

        img_h, img_w = gray.shape[:2]

        max_area_px = (
            img_h * img_w
        ) * 0.9

        # -----------------------------
        # Convert contours → polygons
        # -----------------------------

        for cnt in contours:

            area_px = cv2.contourArea(cnt)

            if (
                area_px > 300
                and area_px <= max_area_px
            ):

                epsilon = (
                    0.005
                    *
                    cv2.arcLength(
                        cnt,
                        True
                    )
                )

                approx = cv2.approxPolyDP(
                    cnt,
                    epsilon,
                    True
                )

                pts = (
                    approx.reshape(-1, 2)
                    /
                    pixels_per_cm
                )

                if len(pts) >= 3:

                    poly = Polygon(pts)

                    if (
                        poly.is_valid
                        and
                        poly.area > 0
                    ):

                        minx, miny, _, _ = (
                            poly.bounds
                        )

                        poly_zeroed = translate(
                            poly,
                            xoff=-minx,
                            yoff=-miny
                        )

                        extracted_polygons.append(
                            poly_zeroed
                        )

        return extracted_polygons

    except Exception:

        return []


# ============================================================
# SVG PREVIEW FUNCTION
# ============================================================

def generate_svg_3pcs_preview(
    p1,
    p2,
    p3,
    width_cm=50,
    height_cm=40
):

    scale = 10

    svg_w = width_cm * scale
    svg_h = height_cm * scale

    svg_code = f'''
    <svg
        width="100%"
        height="auto"
        viewBox="0 0 {svg_w} {svg_h}"
        xmlns="http://www.w3.org/2000/svg"
        style="
            background-color:#F8F9FA;
            border:2px dashed #666;
            border-radius:8px;
        "
    >
    '''

    items = [
        (p1, "#3388ff"),
        (p2, "#ff4444"),
        (p3, "#28a745")
    ]

    for poly, col in items:

        pts = list(
            poly.exterior.coords
        )

        pts_str = " ".join(
            [
                f"{p[0] * scale:.1f},"
                f"{p[1] * scale:.1f}"
                for p in pts
            ]
        )

        svg_code += f'''
        <polygon
            points="{pts_str}"
            fill="{col}"
            stroke="#111"
            stroke-width="1"
            opacity="0.85"
        />
        '''

    svg_code += "</svg>"

    return svg_code


# ============================================================
# COLLISION FUNCTION
# ============================================================

def polygons_collide(
    poly_a,
    poly_b,
    gap=0.1
):

    safe_a = poly_a.buffer(
        gap / 2
    )

    safe_b = poly_b.buffer(
        gap / 2
    )

    return safe_a.intersects(
        safe_b
    )


# ============================================================
# SHEET BOUNDARY FUNCTION
# ============================================================

def polygon_inside_sheet(
    poly,
    sheet_width,
    sheet_length,
    margin
):

    minx, miny, maxx, maxy = (
        poly.bounds
    )

    return (
        minx >= margin
        and
        miny >= margin
        and
        maxx <= sheet_width - margin
        and
        maxy <= sheet_length - margin
    )


# ============================================================
# MAIN APP
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Gambar Pattern Component Master",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


if uploaded_file is not None:

    file_bytes = uploaded_file.read()

    raw_polygons = (
        extract_polygons_from_bytes(
            file_bytes
        )
    )

    # ========================================================
    # NO POLYGON
    # ========================================================

    if not raw_polygons:

        st.error(
            "Gagal mendeteksi bentuk pola dari gambar. "
            "Pastikan garis kontur pola jelas."
        )

    else:

        # ====================================================
        # BASE PATTERN
        # ====================================================

        base_poly = raw_polygons[0]

        bw = (
            base_poly.bounds[2]
            -
            base_poly.bounds[0]
        )

        bh = (
            base_poly.bounds[3]
            -
            base_poly.bounds[1]
        )

        # ====================================================
        # STEP 1
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🛠️ Step 1: Atur Posisi Master 3 Komponen"
        )

        col_ctrl, col_prev = st.columns(
            [1.1, 0.9]
        )

        # ====================================================
        # CONTROLS
        # ====================================================

        with col_ctrl:

            # ------------------------------------------------
            # COMPONENT 1
            # ------------------------------------------------

            st.markdown(
                "##### 🔵 Komponen 1 (Baris 1 - Pcs 1)"
            )

            c1_1, c1_2, c1_3 = st.columns(3)

            with c1_1:

                rot1 = st.slider(
                    "Rotasi Pcs 1 (°)",
                    0,
                    360,
                    90,
                    step=5
                )

            with c1_2:

                shift_x1 = st.slider(
                    "Geser X Pcs 1",
                    -float(bw * 2),
                    float(bw * 2),
                    0.0,
                    step=0.1
                )

            with c1_3:

                shift_y1 = st.slider(
                    "Geser Y Pcs 1",
                    -float(bh * 2),
                    float(bh * 2),
                    0.0,
                    step=0.1
                )

            # ------------------------------------------------
            # COMPONENT 2
            # ------------------------------------------------

            st.markdown(
                "##### 🔴 Komponen 2 (Baris 1 - Pcs 2)"
            )

            c2_1, c2_2, c2_3 = st.columns(3)

            with c2_1:

                rot2 = st.slider(
                    "Rotasi Pcs 2 (°)",
                    0,
                    360,
                    270,
                    step=5
                )

            with c2_2:

                shift_x2 = st.slider(
                    "Geser X Pcs 2",
                    -float(bw * 2),
                    float(bw * 2),
                    4.5,
                    step=0.1
                )

            with c2_3:

                shift_y2 = st.slider(
                    "Geser Y Pcs 2",
                    -float(bh * 2),
                    float(bh * 2),
                    -8.5,
                    step=0.1
                )

            # ------------------------------------------------
            # COMPONENT 3
            # ------------------------------------------------

            st.markdown(
                "##### 🟢 Komponen 3 (Awal Baris 2)"
            )

            c3_1, c3_2, c3_3 = st.columns(3)

            with c3_1:

                rot3 = st.slider(
                    "Rotasi Pcs 3 (°)",
                    0,
                    360,
                    rot1,
                    step=5
                )

            with c3_2:

                r2_shift_x = st.slider(
                    "Geser X Pcs 3",
                    -float(bw * 2),
                    float(bw * 2),
                    8.1,
                    step=0.1
                )

            with c3_3:

                r2_shift_y = st.slider(
                    "Geser Y Pcs 3",
                    -float(bh * 2),
                    float(bh * 2),
                    6.1,
                    step=0.1
                )

        # ====================================================
        # CREATE COMPONENT POLYGONS
        # ====================================================

        p1_rot = rotate(
            base_poly,
            rot1,
            origin="center"
        )

        p1_poly = translate(
            p1_rot,
            xoff=shift_x1,
            yoff=shift_y1
        )

        p2_rot = rotate(
            base_poly,
            rot2,
            origin="center"
        )

        p2_poly = translate(
            p2_rot,
            xoff=shift_x2,
            yoff=shift_y2
        )

        p3_rot = rotate(
            base_poly,
            rot3,
            origin="center"
        )

        p3_poly = translate(
            p3_rot,
            xoff=r2_shift_x,
            yoff=r2_shift_y
        )

        # ====================================================
        # NORMALIZE PREVIEW
        # ====================================================

        min_canvas_x = min(
            p1_poly.bounds[0],
            p2_poly.bounds[0],
            p3_poly.bounds[0]
        )

        min_canvas_y = min(
            p1_poly.bounds[1],
            p2_poly.bounds[1],
            p3_poly.bounds[1]
        )

        pad = 5.0

        p1_preview = translate(
            p1_poly,
            xoff=-min_canvas_x + pad,
            yoff=-min_canvas_y + pad
        )

        p2_preview = translate(
            p2_poly,
            xoff=-min_canvas_x + pad,
            yoff=-min_canvas_y + pad
        )

        p3_preview = translate(
            p3_poly,
            xoff=-min_canvas_x + pad,
            yoff=-min_canvas_y + pad
        )

        # ====================================================
        # COLLISION PREVIEW
        # ====================================================

        collide_12 = polygons_collide(
            p1_preview,
            p2_preview,
            inter_gap
        )

        collide_13 = polygons_collide(
            p1_preview,
            p3_preview,
            inter_gap
        )

        collide_23 = polygons_collide(
            p2_preview,
            p3_preview,
            inter_gap
        )

        # ====================================================
        # PREVIEW
        # ====================================================

        with col_prev:

            st.markdown(
                "### 👁️ Preview Master (3 Komponen)"
            )

            if (
                collide_12
                or collide_13
                or collide_23
            ):

                st.error(
                    "⚠️ Terdapat Komponen yang "
                    "Bertabrakan! Adjust slider "
                    "sampai posisi aman."
                )

            else:

                st.success(
                    "✅ 3 Komponen Bebas Tabrakan "
                    "(Layout Safe)"
                )

            pw = (
                max(
                    p1_preview.bounds[2],
                    p2_preview.bounds[2],
                    p3_preview.bounds[2]
                )
                +
                pad
            )

            ph = (
                max(
                    p1_preview.bounds[3],
                    p2_preview.bounds[3],
                    p3_preview.bounds[3]
                )
                +
                pad
            )

            svg_3pcs = (
                generate_svg_3pcs_preview(
                    p1_preview,
                    p2_preview,
                    p3_preview,
                    width_cm=max(
                        pw,
                        25
                    ),
                    height_cm=max(
                        ph,
                        25
                    )
                )
            )

            st.components.v1.html(
                svg_3pcs,
                height=380,
                scrolling=False
            )

        # ====================================================
        # STEP 2
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🚀 Step 2: Duplikasi Ke Lembaran Utuh"
        )

        if st.button(
            "📊 Duplikasi & Render Full Sheet Layout",
            type="primary"
        ):

            # =================================================
            # INITIAL VARIABLES
            # =================================================

            placed_polygons = []

            total_pattern_area = 0.0

            total_items = (
                target_pairs * 2
            )

            item_idx = 0
            row_idx = 0

            # =================================================
            # NORMALIZE MASTER
            # =================================================

            all_master = [
                p1_poly,
                p2_poly,
                p3_poly
            ]

            base_x0 = min(
                p.bounds[0]
                for p in all_master
            )

            base_y0 = min(
                p.bounds[1]
                for p in all_master
            )

            p1_m = translate(
                p1_poly,
                xoff=-base_x0,
                yoff=-base_y0
            )

            p2_m = translate(
                p2_poly,
                xoff=-base_x0,
                yoff=-base_y0
            )

            p3_m = translate(
                p3_poly,
                xoff=-base_x0,
                yoff=-base_y0
            )

            # =================================================
            # VALIDATE MASTER
            # =================================================

            master_polys = [
                p1_m,
                p2_m,
                p3_m
            ]

            master_collision = False

            for i in range(
                len(master_polys)
            ):

                for j in range(
                    i + 1,
                    len(master_polys)
                ):

                    if polygons_collide(
                        master_polys[i],
                        master_polys[j],
                        inter_gap
                    ):

                        master_collision = True

            if master_collision:

                st.error(
                    "⚠️ Master layout P1/P2/P3 "
                    "masih bertabrakan. "
                    "Silakan adjust slider Step 1."
                )

                st.stop()

            # =================================================
            # PAIR WIDTH
            # =================================================

            pair_min_x = min(
                p1_m.bounds[0],
                p2_m.bounds[0]
            )

            pair_max_x = max(
                p1_m.bounds[2],
                p2_m.bounds[2]
            )

            pair_width = (
                pair_max_x
                -
                pair_min_x
            )

            step_x = (
                pair_width
                +
                inter_gap
            )

            # =================================================
            # ROW OFFSET
            # =================================================

            row_off_x = (
                p3_m.bounds[0]
                -
                p1_m.bounds[0]
            )

            row_off_y = (
                p3_m.bounds[1]
                -
                p1_m.bounds[1]
            )

            # Safety fallback
            if abs(row_off_y) < 0.01:

                row_off_y = (
                    max(
                        p1_m.bounds[3],
                        p2_m.bounds[3],
                        p3_m.bounds[3]
                    )
                    +
                    inter_gap
                )

            # =================================================
            # PLACE ROWS
            # =================================================

            max_rows = 1000

            while (
                item_idx < total_items
                and
                row_idx < max_rows
            ):

                is_row_even = (
                    row_idx % 2 == 1
                )

                row_y = (
                    margin
                    +
                    row_idx * row_off_y
                )

                x_shift = (
                    row_off_x
                    if is_row_even
                    else 0.0
                )

                # ---------------------------------------------
                # Stop if row is below sheet
                # ---------------------------------------------

                row_min_y = min(
                    p1_m.bounds[1],
                    p2_m.bounds[1]
                )

                row_max_y = max(
                    p1_m.bounds[3],
                    p2_m.bounds[3]
                )

                if (
                    row_y + row_max_y
                    >
                    sheet_length - margin
                ):

                    break

                # ---------------------------------------------
                # Starting X
                # ---------------------------------------------

                curr_x = (
                    margin
                    +
                    x_shift
                )

                while curr_x < margin:

                    curr_x += step_x

                # ---------------------------------------------
                # Place pairs
                # ---------------------------------------------

                while (
                    item_idx < total_items
                ):

                    candidate_p1 = translate(
                        p1_m,
                        xoff=curr_x,
                        yoff=row_y
                    )

                    candidate_p2 = translate(
                        p2_m,
                        xoff=curr_x,
                        yoff=row_y
                    )

                    # =========================================
                    # BOUNDARY CHECK
                    # =========================================

                    p1_inside = (
                        polygon_inside_sheet(
                            candidate_p1,
                            sheet_width,
                            sheet_length,
                            margin
                        )
                    )

                    p2_inside = (
                        polygon_inside_sheet(
                            candidate_p2,
                            sheet_width,
                            sheet_length,
                            margin
                        )
                    )

                    if not (
                        p1_inside
                        and
                        p2_inside
                    ):

                        break

                    # =========================================
                    # P1 vs P2
                    # =========================================

                    pair_collision = (
                        polygons_collide(
                            candidate_p1,
                            candidate_p2,
                            inter_gap
                        )
                    )

                    if pair_collision:

                        curr_x += (
                            max(
                                inter_gap,
                                0.1
                            )
                        )

                        continue

                    # =========================================
                    # CHECK AGAINST ALL EXISTING POLYGONS
                    # =========================================

                    p1_collision = False
                    p2_collision = False

                    for old_poly, _ in placed_polygons:

                        if polygons_collide(
                            candidate_p1,
                            old_poly,
                            inter_gap
                        ):

                            p1_collision = True
                            break

                    if not p1_collision:

                        for old_poly, _ in placed_polygons:

                            if polygons_collide(
                                candidate_p2,
                                old_poly,
                                inter_gap
                            ):

                                p2_collision = True
                                break

                    # =========================================
                    # IF COLLISION → MOVE RIGHT
                    # =========================================

                    if (
                        p1_collision
                        or
                        p2_collision
                    ):

                        curr_x += (
                            max(
                                inter_gap,
                                0.1
                            )
                        )

                        continue

                    # =========================================
                    # ACCEPT P1
                    # =========================================

                    placed_polygons.append(
                        (
                            candidate_p1,
                            0
                        )
                    )

                    total_pattern_area += (
                        candidate_p1.area
                    )

                    item_idx += 1

                    # =========================================
                    # ACCEPT P2
                    # =========================================

                    if (
                        item_idx
                        <
                        total_items
                    ):

                        placed_polygons.append(
                            (
                                candidate_p2,
                                1
                            )
                        )

                        total_pattern_area += (
                            candidate_p2.area
                        )

                        item_idx += 1

                    # =========================================
                    # NEXT POSITION
                    # =========================================

                    curr_x += step_x

                row_idx += 1

            # =================================================
            # RESULT VALIDATION
            # =================================================

            if item_idx < total_items:

                st.warning(
                    f"⚠️ Hanya {item_idx} pcs yang "
                    f"berhasil ditempatkan dari target "
                    f"{total_items} pcs."
                )

            else:

                st.success(
                    f"✅ Semua {total_items} pcs "
                    f"berhasil ditempatkan tanpa collision."
                )

            # =================================================
            # SHEET AREA
            # =================================================

            total_sheet_area = (
                sheet_width
                *
                sheet_length
            )

            # =================================================
            # USED LENGTH
            # =================================================

            if placed_polygons:

                max_used_y = max(
                    poly.bounds[3]
                    for poly, _ in placed_polygons
                )

            else:

                max_used_y = 0.0

            # =================================================
            # USED AREA
            # =================================================

            if max_used_y > 0:

                used_sheet_area = (
                    sheet_width
                    *
                    max_used_y
                )

            else:

                used_sheet_area = (
                    total_sheet_area
                )

            # =================================================
            # YIELD
            # =================================================

            component_yield = (

                total_pattern_area
                /
                used_sheet_area
                *
                100

                if used_sheet_area > 0

                else 0.0
            )

            overall_sheet_yield = (

                total_pattern_area
                /
                total_sheet_area
                *
                100

                if total_sheet_area > 0

                else 0.0
            )

            total_waste = (
                100.0
                -
                component_yield
            )

            # =================================================
            # PAIRS
            # =================================================

            pairs_completed = (
                len(placed_polygons)
                //
                2
            )

            # =================================================
            # CONSUMPTION
            # =================================================

            consumption_per_pair = (

                (
                    used_sheet_area
                    /
                    10000
                )
                /
                max(
                    pairs_completed,
                    1
                )
            )

            # =================================================
            # SUMMARY
            # =================================================

            st.markdown(
                "### 📊 Yield & Material Consumption Summary"
            )

            m1, m2, m3, m4, m5 = (
                st.columns(5)
            )

            m1.metric(
                "Komponen Terpasang",
                f"{len(placed_polygons)} pcs "
                f"({pairs_completed} pairs)"
            )

            m2.metric(
                "Total Net Area",
                f"{total_pattern_area:.1f} cm²"
            )

            m3.metric(
                "Component Yield",
                f"{component_yield:.2f} %"
            )

            m4.metric(
                "Overall Sheet Yield",
                f"{overall_sheet_yield:.2f} %"
            )

            m5.metric(
                "Cutting Waste",
                f"{total_waste:.2f} %"
            )

            st.info(
                f"💡 **Consumption Rate:** "
                f"{consumption_per_pair:.4f} m² / pair "
                f"| Panjang Bahan Terpakai: "
                f"{max_used_y:.1f} cm "
                f"dari {sheet_length:.1f} cm"
            )

            # =================================================
            # SVG FULL SHEET
            # =================================================

            scale_f = 8

            svg_w_f = (
                sheet_width
                *
                scale_f
            )

            svg_h_f = (
                sheet_length
                *
                scale_f
            )

            svg_full = f'''
            <svg
                width="100%"
                height="auto"
                viewBox="0 0 {svg_w_f} {svg_h_f}"
                xmlns="http://www.w3.org/2000/svg"
                style="
                    background-color:#F8F9FA;
                    border:2px solid #333;
                    border-radius:8px;
                "
            >
            '''

            # =================================================
            # MARGIN BOX
            # =================================================

            m_x = (
                margin
                *
                scale_f
            )

            m_y = (
                margin
                *
                scale_f
            )

            m_w = (
                sheet_width
                -
                2 * margin
            ) * scale_f

            m_h = (
                sheet_length
                -
                2 * margin
            ) * scale_f

            svg_full += f'''
            <rect
                x="{m_x}"
                y="{m_y}"
                width="{m_w}"
                height="{m_h}"
                fill="none"
                stroke="#ff4444"
                stroke-dasharray="4"
                stroke-width="1.5"
            />
            '''

            # =================================================
            # USED LENGTH LINE
            # =================================================

            if max_used_y > 0:

                c_y = (
                    max_used_y
                    *
                    scale_f
                )

                svg_full += f'''
                <line
                    x1="0"
                    y1="{c_y}"
                    x2="{svg_w_f}"
                    y2="{c_y}"
                    stroke="#3388ff"
                    stroke-dasharray="3"
                    stroke-width="2"
                />
                '''

            # =================================================
            # DRAW POLYGONS
            # =================================================

            colors = [
                "#3388ff",
                "#ff4444"
            ]

            for poly, idx in placed_polygons:

                pts = list(
                    poly.exterior.coords
                )

                pts_str = " ".join(
                    [
                        f"{p[0] * scale_f:.1f},"
                        f"{p[1] * scale_f:.1f}"
                        for p in pts
                    ]
                )

                fill_col = colors[
                    idx % 2
                ]

                svg_full += f'''
                <polygon
                    points="{pts_str}"
                    fill="{fill_col}"
                    stroke="#111"
                    stroke-width="0.8"
                    opacity="0.85"
                />
                '''

            svg_full += "</svg>"

            # =================================================
            # DISPLAY
            # =================================================

            st.components.v1.html(
                svg_full,
                height=650,
                scrolling=True
            )
