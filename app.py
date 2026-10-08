import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Footwear Material Yield Visualizer", page_icon="📐", layout="wide"
)

st.title("⚡ Footwear Material Nesting & Drawing Studio")
st.markdown("---")

# ============================================================
# TAHAP 1: SETUP PARAMETER UTAMA DI HALAMAN UTAMA (COMPACT GRID)
# ============================================================

st.subheader("📋 Material Setup")
col_p1, col_p1_unit, col_p2, col_p3, col_p4 = st.columns(
    [1.5, 1, 1.2, 1.2, 1.2]
)

with col_p1:
  material_width_input = st.number_input(
      "Material Width", value=44.0, step=1.0, format="%.2f"
  )
with col_p1_unit:
  width_unit = st.selectbox("Width Unit", ["inch", "cm", "m"], index=0)

# Konversi lebar material ke cm berdasarkan pilihan unit
if width_unit == "inch":
  sheet_width = material_width_input * 2.54
elif width_unit == "m":
  sheet_width = material_width_input * 100.0
else:
  sheet_width = material_width_input

with col_p2:
  sheet_length = st.number_input("Material Length (cm)", value=100.0, step=5.0)
with col_p3:
  margin = st.number_input("Margin (cm)", value=1.0, step=0.5)
with col_p4:
  target_pairs = st.number_input("Target Pairs", value=50, min_value=1, step=1)
  target_pieces = target_pairs * 2

st.markdown("---")
st.subheader("📥 Input Sumber Pola Komponen")

# Pilihan metode input: Upload File vs Kanvas Gambar Langsung
input_method = st.radio(
    "Pilih Cara Input Pola:",
    (
        "Upload Gambar Pola (File)",
        "Gambar Langsung di Kanvas (Interactive Canvas)",
    ),
    horizontal=True,
)

file_bytes = None

if input_method == "Upload Gambar Pola (File)":
  uploaded_file = st.file_uploader(
      "Upload Pattern Component Master Image", type=["png", "jpg", "jpeg"]
  )
  if uploaded_file is not None:
    file_bytes = uploaded_file.read()

else:
  st.markdown(
      "Gunakan kanvas di bawah untuk menggambar atau membuat sketsa bentuk"
      " komponen sepatu secara manual:"
  )

  col_c1, col_c2, col_c3 = st.columns(3)
  with col_c1:
    canvas_mode = st.selectbox(
        "Mode Gambar Kanvas",
        ("polygon", "rect", "circle", "freedraw", "transform"),
    )
  with col_c2:
    canvas_stroke_width = st.slider("Ketebalan Garis", 1, 10, 2)
  with col_c3:
    canvas_stroke_color = st.color_picker("Wрна Garis", "#000000")

  # Komponen St Canvas
  canvas_result = st_canvas(
      fill_color="rgba(51, 136, 255, 0.3)",
      stroke_width=canvas_stroke_width,
      stroke_color=canvas_stroke_color,
      background_color="#FFFFFF",
      update_streamlit=True,
      height=400,
      width=700,
      drawing_mode=canvas_mode,
      key="canvas_pola_sepatu",
  )

  # Jika user menggambar sesuatu di kanvas, ubah objek kanvas menjadi gambar byte (PNG)
  if canvas_result.image_data is not None:
    # Konversi array RGBA canvas ke format gambar OpenCV/PNG bytes
    canvas_img = canvas_result.image_data.astype(np.uint8)
    success, encoded_img = cv2.imencode(".png", canvas_img)
    if success:
      file_bytes = encoded_img.tobytes()

# ============================================================
# EXTRACT POLYGONS FROM IMAGE / CANVAS
# ============================================================


@st.cache_data
def extract_polygons_from_bytes(file_bytes, dpi=96):
  try:
    img = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED
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
        2,
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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


def generate_svg_preview_grid(items_with_color, width_cm=60, height_cm=40):
  scale = 4.5
  svg_w = width_cm * scale
  svg_h = height_cm * scale

  svg_code = (
      f'<svg width="100%" height="auto" viewBox="0 0 {svg_w} {svg_h}"'
      ' xmlns="http://www.w3.org/2000/svg"'
      ' style="background-color:#F8F9FA; border:2px dashed #666;'
      ' border-radius:8px;">'
  )

  color_map = {0: "#3388ff", 1: "#ff4444"}
  for poly, color_idx in items_with_color:
    pts = list(poly.exterior.coords)
    pts_str = " ".join([f"{p[0] * scale:.2f},{p[1] * scale:.2f}" for p in pts])
    col = color_map.get(color_idx % 2, "#3388ff")
    svg_code += (
        f'<polygon points="{pts_str}" fill="{col}" stroke="#111"'
        ' stroke-width="0.8" opacity="0.85"/>'
    )

  svg_code += "</svg>"
  return svg_code


# ============================================================
# MAIN APP LOGIC (JIKA FILE ATAU KANVAS TERSEDIA)
# ============================================================

if file_bytes is not None:
  raw_polygons = extract_polygons_from_bytes(file_bytes)

  if not raw_polygons:
    st.warning(
        "⏳ Belum ada pola terbaca. Silakan unggah gambar yang jelas atau"
        " gambar bentuk komponen pada kanvas di atas."
    )
  else:
    base_poly = raw_polygons[0]
    bw = base_poly.bounds[2] - base_poly.bounds[0]
    bh = base_poly.bounds[3] - base_poly.bounds[1]

    st.markdown("---")
    st.subheader("🛠️ Layout Configuration")

    col_ctrl, col_prev = st.columns([1.1, 0.9])

    with col_ctrl:
      category = st.selectbox(
          "Select Layout Category",
          [
              "Category 1: One Way Straight",
              "Category 2: Two Way Interlock",
              "Category 3: One Way Staggered",
              "Category 4: Two Way Staggered",
          ],
      )

      is_twoway_cat = "Two Way" in category
      is_staggered_cat = "Staggered" in category

      c_rot1, c_rot2 = st.columns(2)
      with c_rot1:
        rot_p1 = st.number_input(
            "Rotation Pcs 1 (°)", min_value=0, max_value=360, value=90, step=5
        )
      with c_rot2:
        if is_twoway_cat:
          default_rot2 = (rot_p1 + 180) % 360
          rot_p2 = st.number_input(
              "Rotation Pcs 2 (°)",
              min_value=0,
              max_value=360,
              value=int(default_rot2),
              step=5,
          )
        else:
          rot_p2 = rot_p1

      st.markdown("##### 🎛️ Spacing Adjustments")
      c_ft1, c_ft2 = st.columns(2)
      with c_ft1:
        fine_tune_x = st.number_input(
            "Column Gap / Step X", value=0.0, step=0.1, format="%.2f"
        )
      with c_ft2:
        fine_tune_y = st.number_input(
            "Row Gap / Pitch Y", value=0.0, step=0.1, format="%.2f"
        )

      if is_staggered_cat:
        preview_shift_x = st.number_input(
            "Row 2 Shift (Offset X)", value=0.0, step=0.1, format="%.2f"
        )
      else:
        preview_shift_x = 0.0

    # HITUNG GEOMETRI P1 & P2
    p1 = rotate(base_poly, rot_p1, origin="center")
    p1 = translate(p1, xoff=-p1.bounds[0], yoff=-p1.bounds[1])

    p2 = rotate(base_poly, rot_p2, origin="center")
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

    else:  # Category 4
      default_stagger_x = step_x / 2
      preview_items.append((p1, 0))
      preview_items.append((translate(p2, xoff=step_x, yoff=0), 1))
      row2_y = pitch_y
      row2_x1 = default_stagger_x + preview_shift_x
      row2_x2 = row2_x1 + step_x
      preview_items.append((translate(p1, xoff=row2_x1, yoff=row2_y), 0))
      preview_items.append((translate(p2, xoff=row2_x2, yoff=row2_y), 1))

    with col_prev:
      st.markdown("##### 👁️ Live Preview Grid (2 Rows)")
      st.info(
          f"💡 Mode: **{category.split(':')[0]}** | Step X: {step_x:.2f} cm |"
          f" Pitch Y: {pitch_y:.2f} cm"
      )
      svg_preview = generate_svg_preview_grid(
          preview_items, width_cm=60, height_cm=40
      )
      st.components.v1.html(svg_preview, height=320, scrolling=False)

    # ============================================================
    # TAHAP 2: RENDER HASIL PENUH KE LEMBARAN BAHAN
    # ============================================================

    st.markdown("---")
    st.subheader("🚀 Nesting Layout")

    if st.button("📊 Process Layout", type="primary", use_container_width=True):
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
          while (
              item_idx < total_items
              and (curr_x + p1_w) <= (sheet_width - margin)
          ):
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

            if (
                curr_x >= margin
                and (curr_x + current_w) <= (sheet_width - margin)
            ):
              placed_polygons.append((cand, col_idx % 2))
              total_pattern_area += cand.area
              item_idx += 1
            curr_x += step_x
            col_idx += 1
          row_idx += 1

      elif "Category 3" in category:
        stagger_x = (step_x / 2) + preview_shift_x
        while item_idx < total_items:
          is_row_even = row_idx % 2 == 1
          row_y = margin + (row_idx * pitch_y)
          if row_y + p1_h > (sheet_length - margin):
            break
          row_start_x = margin + (stagger_x if is_row_even else 0.0)
          while row_start_x - step_x >= margin:
            row_start_x -= step_x
          curr_x = row_start_x
          col_idx = 0
          while item_idx < total_items and curr_x <= (sheet_width - margin):
            if (
                curr_x >= margin
                and (curr_x + p1_w) <= (sheet_width - margin)
            ):
              cand = translate(p1, xoff=curr_x, yoff=row_y)
              placed_polygons.append((cand, col_idx % 2))
              total_pattern_area += cand.area
              item_idx += 1
            curr_x += step_x
            col_idx += 1
          row_idx += 1

      else:  # Category 4
        h_unit = max(p1_h, p2_h)
        stagger_x = (step_x / 2) + preview_shift_x
        while item_idx < total_items:
          is_row_even = row_idx % 2 == 1
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

            if curr_x >= margin and (curr_x + current_w) <= (
                sheet_width - margin
            ):
              placed_polygons.append((cand, col_idx % 2))
              total_pattern_area += cand.area
              item_idx += 1
            curr_x += step_x
            col_idx += 1
          row_idx += 1

      # ============================================================
      # PERHITUNGAN STANDAR METRIK (CLEAN FORMATTING & YIELD)
      # ============================================================
      pieces_completed = len(placed_polygons)
      pairs_completed = max(pieces_completed // 2, 1)

      single_net_area = base_poly.area
      net_area_per_pair = single_net_area * 2.0

      max_used_y = (
          max([p.bounds[3] for p, _ in placed_polygons])
          if placed_polygons
          else sheet_length
      )

      used_sheet_area = sheet_width * max_used_y
      gross_area_per_pair = (
          used_sheet_area / pairs_completed if pairs_completed > 0 else 0.0
      )
      waste_area_per_pair = gross_area_per_pair - net_area_per_pair
      efficiency = (
          (net_area_per_pair / gross_area_per_pair) * 100
          if gross_area_per_pair > 0
          else 0.0
      )

      used_length_m = max_used_y / 100.0
      yield_value = (
          used_length_m / pairs_completed if pairs_completed > 0 else 0.0
      )

      st.markdown("### 📊 Summary Report")

      col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
      col_m1.metric("Parts per pair", f"{2.00:.2f}")
      col_m2.metric("Net Area / pair", f"{net_area_per_pair:.4f}")
      col_m3.metric("Gross Area / pair", f"{gross_area_per_pair:.4f}")
      col_m4.metric("Waste Area / pair", f"{waste_area_per_pair:.4f}")
      col_m5.metric("Efficiency (%)", f"{efficiency:.2f}%")
      col_m6.metric("Yield (per meter)", f"{yield_value:.4f}")

      st.info(
          f"💡 **Production Info:** Placed {pairs_completed} pairs"
          f" ({pieces_completed} pcs) | Used Length: **{max_used_y:.2f} cm**"
          f" out of {sheet_length:.1f} cm (Width: {sheet_width:.1f} cm)"
      )

      # RENDER SVG FULL SHEET
      scale_f = 6.0
      svg_w_f = sheet_width * scale_f
      svg_h_f = sheet_length * scale_f

      svg_full = (
          f'<svg width="100%" height="auto" viewBox="0 0 {svg_w_f} {svg_h_f}"'
          ' xmlns="http://www.w3.org/2000/svg" style="background-color:'
          ' #F8F9FA; border: 2px solid #333; border-radius: 8px;">'
      )

      m_x = margin * scale_f
      m_y = margin * scale_f
      m_w = (sheet_width - 2 * margin) * scale_f
      m_h = (sheet_length - 2 * margin) * scale_f
      svg_full += (
          f'<rect x="{m_x}" y="{m_y}" width="{m_w}" height="{m_h}" fill="none"'
          ' stroke="#ff4444" stroke-dasharray="4" stroke-width="1.5"/>'
      )

      if max_used_y > 0:
        c_y = max_used_y * scale_f
        svg_full += (
            f'<line x1="0" y1="{c_y}" x2="{svg_w_f}" y2="{c_y}"'
            ' stroke="#3388ff" stroke-dasharray="3" stroke-width="2"/>'
        )

      for poly, idx in placed_polygons:
        pts = list(poly.exterior.coords)
        pts_str = " ".join(
            [f"{p[0] * scale_f:.2f},{p[1] * scale_f:.2f}" for p in pts]
        )
        fill_col = "#3388ff" if idx % 2 == 0 else "#ff4444"
        svg_full += (
            f'<polygon points="{pts_str}" fill="{fill_col}" stroke="#111"'
            ' stroke-width="0.6" opacity="0.85"/>'
        )

      svg_full += "</svg>"
      st.components.v1.html(svg_full, height=650, scrolling=True)
