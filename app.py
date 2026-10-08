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

       # ============================================================
# STEP 2: TIGHT NESTING / MATERIAL OPTIMIZATION
# ============================================================

st.markdown("---")
st.subheader("🚀 Step 2: Tight Nesting Ke Lembaran Utuh")

if st.button(
    "📊 Generate Tight Nesting",
    type="primary"
):

    # ========================================================
    # BASIC SETTINGS
    # ========================================================

    total_items = target_pairs * 2

    placed_polygons = []

    total_pattern_area = 0.0

    item_idx = 0
    row_idx = 0

    # Resolusi pencarian.
    # Semakin kecil → semakin rapat tetapi lebih lambat.
    search_step = 0.10

    # ========================================================
    # NORMALIZE MASTER
    # ========================================================

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

    # ========================================================
    # MASTER COLLISION CHECK
    # ========================================================

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
            "⚠️ Master P1/P2/P3 masih bertabrakan. "
            "Adjust Step 1 terlebih dahulu."
        )

        st.stop()

    # ========================================================
    # MASTER DIMENSION
    # ========================================================

    master_min_x = min(
        p.bounds[0]
        for p in master_polys
    )

    master_max_x = max(
        p.bounds[2]
        for p in master_polys
    )

    master_min_y = min(
        p.bounds[1]
        for p in master_polys
    )

    master_max_y = max(
        p.bounds[3]
        for p in master_polys
    )

    master_width = (
        master_max_x
        -
        master_min_x
    )

    master_height = (
        master_max_y
        -
        master_min_y
    )

    # ========================================================
    # ROW OFFSET
    # ========================================================

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

    # Safety
    if abs(row_off_y) < 0.01:

        row_off_y = (
            master_height
            +
            inter_gap
        )

    # ========================================================
    # COLLISION CHECK
    # ========================================================

    def candidate_is_safe(
        candidates,
        existing
    ):

        # ----------------------------------------------
        # Check candidate vs candidate
        # ----------------------------------------------

        for i in range(
            len(candidates)
        ):

            for j in range(
                i + 1,
                len(candidates)
            ):

                if polygons_collide(
                    candidates[i],
                    candidates[j],
                    inter_gap
                ):

                    return False

        # ----------------------------------------------
        # Check candidate vs existing
        # ----------------------------------------------

        for candidate in candidates:

            for old_poly, _ in existing:

                if polygons_collide(
                    candidate,
                    old_poly,
                    inter_gap
                ):

                    return False

        return True

    # ========================================================
    # SHEET CHECK
    # ========================================================

    def candidates_inside_sheet(
        candidates
    ):

        for poly in candidates:

            if not polygon_inside_sheet(
                poly,
                sheet_width,
                sheet_length,
                margin
            ):

                return False

        return True

    # ========================================================
    # FIND TIGHT X
    # ========================================================

    def find_tight_x(
        base_x,
        row_y,
        existing
    ):

        # ----------------------------------------------------
        # Start from normal position
        # ----------------------------------------------------

        x = base_x

        candidates = [
            translate(
                p1_m,
                xoff=x,
                yoff=row_y
            ),
            translate(
                p2_m,
                xoff=x,
                yoff=row_y
            )
        ]

        # ----------------------------------------------------
        # Move left until collision
        # ----------------------------------------------------

        last_safe_x = None

        while True:

            test_x = (
                x
                -
                search_step
            )

            test_candidates = [

                translate(
                    p1_m,
                    xoff=test_x,
                    yoff=row_y
                ),

                translate(
                    p2_m,
                    xoff=test_x,
                    yoff=row_y
                )
            ]

            # ----------------------------------------------
            # Sheet boundary
            # ----------------------------------------------

            if not candidates_inside_sheet(
                test_candidates
            ):

                break

            # ----------------------------------------------
            # Collision
            # ----------------------------------------------

            if not candidate_is_safe(
                test_candidates,
                existing
            ):

                break

            x = test_x

            last_safe_x = x

        # ----------------------------------------------------
        # If no left movement was possible
        # use original position
        # ----------------------------------------------------

        if last_safe_x is None:

            return x

        return last_safe_x

    # ========================================================
    # FIND TIGHT Y
    # ========================================================

    def find_tight_y(
        base_y,
        row_x,
        existing
    ):

        y = base_y

        last_safe_y = y

        while True:

            test_y = (
                y
                -
                search_step
            )

            test_candidates = [

                translate(
                    p1_m,
                    xoff=row_x,
                    yoff=test_y
                ),

                translate(
                    p2_m,
                    xoff=row_x,
                    yoff=test_y
                )
            ]

            if not candidates_inside_sheet(
                test_candidates
            ):

                break

            if not candidate_is_safe(
                test_candidates,
                existing
            ):

                break

            y = test_y

            last_safe_y = y

        return last_safe_y

    # ========================================================
    # BUILD ROWS
    # ========================================================

    max_rows = 1000

    while (
        item_idx < total_items
        and
        row_idx < max_rows
    ):

        # ====================================================
        # NOMINAL ROW POSITION
        # ====================================================

        row_y = (
            margin
            +
            row_idx * row_off_y
        )

        # Alternating row offset
        if row_idx % 2 == 1:

            row_x = (
                margin
                +
                row_off_x
            )

        else:

            row_x = margin

        # ====================================================
        # TIGHTEN ROW VERTICALLY
        # ====================================================

        if placed_polygons:

            row_y = find_tight_y(
                row_y,
                row_x,
                placed_polygons
            )

        # ====================================================
        # START X
        # ====================================================

        curr_x = row_x

        # ====================================================
        # PLACE PAIRS
        # ====================================================

        while (
            item_idx < total_items
        ):

            # -----------------------------------------------
            # Create pair
            # -----------------------------------------------

            candidates = [

                translate(
                    p1_m,
                    xoff=curr_x,
                    yoff=row_y
                ),

                translate(
                    p2_m,
                    xoff=curr_x,
                    yoff=row_y
                )

            ]

            # -----------------------------------------------
            # Boundary
            # -----------------------------------------------

            if not candidates_inside_sheet(
                candidates
            ):

                break

            # -----------------------------------------------
            # Collision
            # -----------------------------------------------

            if not candidate_is_safe(
                candidates,
                placed_polygons
            ):

                # Move right
                curr_x += search_step

                continue

            # =================================================
            # TIGHTEN HORIZONTALLY
            # =================================================

            tight_x = find_tight_x(
                curr_x,
                row_y,
                placed_polygons
            )

            candidates = [

                translate(
                    p1_m,
                    xoff=tight_x,
                    yoff=row_y
                ),

                translate(
                    p2_m,
                    xoff=tight_x,
                    yoff=row_y
                )

            ]

            # =================================================
            # FINAL VALIDATION
            # =================================================

            if not candidates_inside_sheet(
                candidates
            ):

                break

            if not candidate_is_safe(
                candidates,
                placed_polygons
            ):

                curr_x += search_step

                continue

            # =================================================
            # ACCEPT P1
            # =================================================

            placed_polygons.append(
                (
                    candidates[0],
                    0
                )
            )

            total_pattern_area += (
                candidates[0].area
            )

            item_idx += 1

            # =================================================
            # ACCEPT P2
            # =================================================

            if (
                item_idx
                <
                total_items
            ):

                placed_polygons.append(
                    (
                        candidates[1],
                        1
                    )
                )

                total_pattern_area += (
                    candidates[1].area
                )

                item_idx += 1

            # =================================================
            # NEXT PAIR
            # =================================================

            pair_max_x = max(
                candidates[0].bounds[2],
                candidates[1].bounds[2]
            )

            # Start next pair from current right edge,
            # NOT from pair_width.
            curr_x = (
                pair_max_x
                -
                inter_gap
            )

            # Small move to avoid infinite loop
            curr_x += search_step

        # ====================================================
        # NEXT ROW
        # ====================================================

        row_idx += 1

    # ========================================================
    # RESULT
    # ========================================================

    if item_idx < total_items:

        st.warning(
            f"⚠️ {item_idx} pcs berhasil ditempatkan "
            f"dari target {total_items} pcs."
        )

    else:

        st.success(
            f"✅ Semua {total_items} pcs berhasil "
            f"ditempatkan dengan tight nesting."
        )

    # ========================================================
    # AREA
    # ========================================================

    total_sheet_area = (
        sheet_width
        *
        sheet_length
    )

    if placed_polygons:

        min_used_x = min(
            poly.bounds[0]
            for poly, _ in placed_polygons
        )

        max_used_x = max(
            poly.bounds[2]
            for poly, _ in placed_polygons
        )

        min_used_y = min(
            poly.bounds[1]
            for poly, _ in placed_polygons
        )

        max_used_y = max(
            poly.bounds[3]
            for poly, _ in placed_polygons
        )

        used_width = (
            max_used_x
            -
            min_used_x
        )

        used_length = (
            max_used_y
            -
            min_used_y
        )

        used_sheet_area = (
            used_width
            *
            used_length
        )

    else:

        used_width = 0
        used_length = 0

        used_sheet_area = 0

    # ========================================================
    # YIELD
    # ========================================================

    component_yield = (

        total_pattern_area
        /
        used_sheet_area
        *
        100

        if used_sheet_area > 0

        else 0
    )

    overall_sheet_yield = (

        total_pattern_area
        /
        total_sheet_area
        *
        100

        if total_sheet_area > 0

        else 0
    )

    total_waste = (
        100
        -
        component_yield
    )

    pairs_completed = (
        len(placed_polygons)
        //
        2
    )

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

    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        "### 📊 Yield & Material Consumption Summary"
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Komponen",
        f"{len(placed_polygons)} pcs"
    )

    m2.metric(
        "Net Area",
        f"{total_pattern_area:.1f} cm²"
    )

    m3.metric(
        "Tight Yield",
        f"{component_yield:.2f}%"
    )

    m4.metric(
        "Sheet Yield",
        f"{overall_sheet_yield:.2f}%"
    )

    m5.metric(
        "Waste",
        f"{total_waste:.2f}%"
    )

    st.info(
        f"💡 **Consumption:** "
        f"{consumption_per_pair:.4f} m² / pair "
        f"| Used Width: {used_width:.1f} cm "
        f"| Used Length: {used_length:.1f} cm"
    )

    # ========================================================
    # SVG
    # ========================================================

    scale_f = 8

    svg_w_f = (
        sheet_width
        *
        scale_f
    )

    svg_h_f = (
        max(
            sheet_length,
            used_length + margin * 2
        )
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

    # ========================================================
    # SHEET
    # ========================================================

    svg_full += f'''
    <rect
        x="0"
        y="0"
        width="{svg_w_f}"
        height="{svg_h_f}"
        fill="#F8F9FA"
    />
    '''

    # ========================================================
    # MARGIN
    # ========================================================

    svg_full += f'''
    <rect
        x="{margin * scale_f}"
        y="{margin * scale_f}"
        width="{(sheet_width - 2 * margin) * scale_f}"
        height="{(sheet_length - 2 * margin) * scale_f}"
        fill="none"
        stroke="#ff4444"
        stroke-dasharray="4"
        stroke-width="1.5"
    />
    '''

    # ========================================================
    # POLYGONS
    # ========================================================

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

    st.components.v1.html(
        svg_full,
        height=700,
        scrolling=True
    )
