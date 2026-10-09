import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Footwear Material Nesting", page_icon="📐", layout="wide"
)

st.title("⚡ Footwear Material Nesting")
st.markdown("---")

# ============================================================
# TAHAP 1: SETUP PARAMETER UTAMA DI HALAMAN UTAMA (COMPACT GRID)
# ============================================================

st.subheader("📋 Material Setup")
col_p1, col_p1_unit, col_p2, col_p3, col_p4, col_p5 = st.columns(
    [1.4, 0.9, 1.1, 1.1, 1.1, 1.2]
)

with col_p1:
  material_width_input = st.number_input(
      "Material Width", value=44.0, step=1.0, format="%.2f"
  )
with col_p1_unit:
  width_unit = st.selectbox("Unit", ["inch", "cm", "m"], index=0)

# Konversi lebar material ke cm berdasarkan pilihan unit
if width_unit == "inch":
  sheet_width = material_width_input * 2.54
elif width_unit == "m":
  sheet_width = material_width_input * 100.0
else:
  sheet_width = material_width_input

with col_p2:
  sheet_length = st.number_input("Length (cm)", value=100.0, step=5.0)
with col_p3:
  margin = st.number_input("Margin (cm)", value=1.0, step=0.5)
with col_p4:
  material_price = st.number_input(
      "Price / Meter ($)", value=5.00, step=0.50, format="%.2f"
  )
with col_p5:
  target_pairs = st.number_input("Target Pairs", value=50, min_value=1, step=1)
  target_pieces = target_pairs * 2

st.markdown("---")
st.subheader("📥 Component Pattern Source Input")

input_method = st.radio(
    "Select Pattern Input Method:",
    ("Upload Pattern Image (File)", "Draw on Interactive Canvas (HTML5 Studio)"),
    horizontal=True,
)

file_bytes = None

if input_method == "Upload Pattern Image (File)":
  uploaded_file = st.file_uploader(
      "Upload Pattern Component Master Image", type=["png", "jpg", "jpeg"]
  )
  if uploaded_file is not None:
    file_bytes = uploaded_file.read()

else:
  st.markdown(
      "💡 **Canvas Guide:** Choose a drawing tool (*Pencil, Line, Rect, Circle,"
      " Polygon, or Eraser*), draw a tightly closed component shape, use"
      " **Undo/Redo** if needed, click **'Download Canvas Image'**, and upload"
      " the resulting file below."
  )

  # Komponen Custom HTML5 Canvas Lengkap dengan Undo, Redo, Polygon, & Download
  canvas_html = """
    <div style="font-family: sans-serif; background: #f9f9f9; padding: 12px; border-radius: 8px; width: fit-content;">
        <div style="margin-bottom: 10px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
            <label style="font-size: 13px; font-weight: bold; color: #333;">Tool:</label>
            <select id="toolSelect" style="padding: 6px; border-radius: 4px; border: 1px solid #ccc; font-weight: bold;">
                <option value="pencil">✏️ Freehand Pencil</option>
                <option value="line">📏 Straight Line</option>
                <option value="rect">⬛ Rectangle / Box</option>
                <option value="circle">⭕ Circle / Oval</option>
                <option value="polygon">📐 Polygon (Multi-click & Double Click to close)</option>
                <option value="eraser">🧹 Eraser</option>
            </select>

            <label style="font-size: 13px; font-weight: bold; color: #333; margin-left: 5px;">Size:</label>
            <input type="range" id="brushSize" min="1" max="15" value="4" style="width: 70px;">

            <button id="undoBtn" style="padding: 6px 12px; background-color: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; margin-left: 10px;">↩️ Undo</button>
            <button id="redoBtn" style="padding: 6px 12px; background-color: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">🔁 Redo</button>
            <button id="clearBtn" style="padding: 6px 12px; background-color: #ff4b4b; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">Clear</button>
            
            <a id="downloadLink" download="footwear_pattern_canvas.png" style="margin-left: auto;">
                <button id="downloadBtn" style="padding: 6px 14px; background-color: #0083B8; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">📥 Download Canvas Image</button>
            </a>
        </div>
        <canvas id="paintCanvas" width="1000" height="800" style="border:2px solid #ccc; background-color:#ffffff; cursor:crosshair; border-radius: 6px; display: block;"></canvas>
        <div id="instruction" style="font-size: 12px; color: #555; margin-top: 6px; font-weight: 500;">Mode: Freehand Pencil - Click and drag to draw freely.</div>
    </div>

    <script>
        const canvas = document.getElementById('paintCanvas');
        const ctx = canvas.getContext('2d');
        let painting = false;
        let startX, startY;
        let snapshot;

        // Undo / Redo history stacks
        let undoStack = [];
        let redoStack = [];
        const maxHistory = 20;

        // Polygon variables
        let polyPoints = [];
        let isDrawingPolygon = false;

        // Set initial white background
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        saveState();

        const toolSelect = document.getElementById('toolSelect');
        const brushSize = document.getElementById('brushSize');
        const instruction = document.getElementById('instruction');

        function saveState() {
            if (undoStack.length >= maxHistory) {
                undoStack.shift();
            }
            undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
            redoStack = []; // Clear redo stack on new action
        }

        toolSelect.addEventListener('change', function() {
            const val = this.value;
            if(val === 'pencil') instruction.innerText = "Mode: Freehand Pencil - Click and drag to draw freely.";
            else if(val === 'line') instruction.innerText = "Mode: Straight Line - Click, drag, and release to draw a straight line.";
            else if(val === 'rect') instruction.innerText = "Mode: Rectangle - Click, drag, and release to draw a rectangle.";
            else if(val === 'circle') instruction.innerText = "Mode: Circle - Click, drag, and release to draw a circle.";
            else if(val === 'polygon') instruction.innerText = "Mode: Polygon - Click consecutive corner points. Double-click to close shape.";
            else if(val === 'eraser') instruction.innerText = "Mode: Eraser - Drag over strokes to erase.";
            polyPoints = [];
            isDrawingPolygon = false;
        });

        function getMousePos(e) {
            const rect = canvas.getBoundingClientRect();
            return {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            };
        }

        canvas.addEventListener('mousedown', (e) => {
            const pos = getMousePos(e);
            const currentTool = toolSelect.value;

            if (currentTool === 'polygon') {
                if (!isDrawingPolygon) {
                    isDrawingPolygon = true;
                    polyPoints = [];
                    snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
                }
                polyPoints.push(pos);
                
                ctx.fillStyle = '#0083B8';
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 3, 0, 2 * Math.PI);
                ctx.fill();

                if (polyPoints.length > 1) {
                    ctx.lineWidth = parseInt(brushSize.value);
                    ctx.strokeStyle = '#000000';
                    ctx.beginPath();
                    ctx.moveTo(polyPoints[polyPoints.length - 2].x, polyPoints[polyPoints.length - 2].y);
                    ctx.lineTo(pos.x, pos.y);
                    ctx.stroke();
                }
                return;
            }

            startX = pos.x;
            startY = pos.y;
            painting = true;
            snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);

            if (currentTool === 'pencil' || currentTool === 'eraser') {
                ctx.beginPath();
                ctx.moveTo(startX, startY);
            }
        });

        canvas.addEventListener('dblclick', (e) => {
            if (toolSelect.value === 'polygon' && isDrawingPolygon && polyPoints.length > 2) {
                ctx.beginPath();
                ctx.moveTo(polyPoints[polyPoints.length - 1].x, polyPoints[polyPoints.length - 1].y);
                ctx.lineTo(polyPoints[0].x, polyPoints[0].y);
                ctx.lineWidth = parseInt(brushSize.value);
                ctx.strokeStyle = '#000000';
                ctx.stroke();

                isDrawingPolygon = false;
                polyPoints = [];
                saveState();
                updateDownloadLink();
            }
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!painting) return;
            const pos = getMousePos(e);
            const currentTool = toolSelect.value;
            const size = parseInt(brushSize.value);

            if (currentTool === 'pencil') {
                ctx.lineWidth = size;
                ctx.lineCap = 'round';
                ctx.strokeStyle = '#000000';
                ctx.lineTo(pos.x, pos.y);
                ctx.stroke();
            } else if (currentTool === 'eraser') {
                ctx.lineWidth = size * 3;
                ctx.lineCap = 'round';
                ctx.strokeStyle = '#ffffff';
                ctx.lineTo(pos.x, pos.y);
                ctx.stroke();
            } else if (currentTool === 'line' || currentTool === 'rect' || currentTool === 'circle') {
                ctx.putImageData(snapshot, 0, 0);
                ctx.lineWidth = size;
                ctx.strokeStyle = '#000000';

                if (currentTool === 'line') {
                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    ctx.lineTo(pos.x, pos.y);
                    ctx.stroke();
                } else if (currentTool === 'rect') {
                    let w = pos.x - startX;
                    let h = pos.y - startY;
                    ctx.strokeRect(startX, startY, w, h);
                } else if (currentTool === 'circle') {
                    let radius = Math.sqrt(Math.pow(pos.x - startX, 2) + Math.pow(pos.y - startY, 2));
                    ctx.beginPath();
                    ctx.arc(startX, startY, radius, 0, 2 * Math.PI);
                    ctx.stroke();
                }
            }
        });

        canvas.addEventListener('mouseup', (e) => {
            if (!painting) return;
            painting = false;
            saveState();
            updateDownloadLink();
        });

        document.getElementById('undoBtn').addEventListener('click', function() {
            if (undoStack.length > 1) {
                redoStack.push(undoStack.pop());
                const prevState = undoStack[undoStack.length - 1];
                ctx.putImageData(prevState, 0, 0);
                updateDownloadLink();
            }
        });

        document.getElementById('redoBtn').addEventListener('click', function() {
            if (redoStack.length > 0) {
                const nextState = redoStack.pop();
                undoStack.push(nextState);
                ctx.putImageData(nextState, 0, 0);
                updateDownloadLink();
            }
        });

        document.getElementById('clearBtn').addEventListener('click', function() {
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            polyPoints = [];
            isDrawingPolygon = false;
            saveState();
            updateDownloadLink();
        });

        function updateDownloadLink() {
            const dataURL = canvas.toDataURL('image/png');
            document.getElementById('downloadLink').href = dataURL;
        }

        updateDownloadLink();
    </script>
    """

  components.html(canvas_html, height=480)

  st.markdown("---")
  uploaded_file = st.file_uploader(
      "📁 Upload Downloaded Canvas PNG File", type=["png", "jpg", "jpeg"]
  )
  if uploaded_file is not None:
    file_bytes = uploaded_file.read()

# ============================================================
# EXTRACT POLYGONS FROM IMAGE
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
      alpha = img[:, :, 3]
      rgb = img[:, :, :3]
      background = np.ones_like(rgb, dtype=np.uint8) * 255
      alpha_factor = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
      gray = cv2.cvtColor(
          (rgb * alpha_factor + background * (1 - alpha_factor)).astype(
              np.uint8
          ),
          cv2.COLOR_BGR2GRAY,
      )
    elif len(img.shape) == 3:
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
      gray = img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    pixels_per_cm = dpi / 2.54
    extracted_polygons = []

    img_h, img_w = gray.shape[:2]
    max_area_px = (img_h * img_w) * 0.9

    for cnt in contours:
      area_px = cv2.contourArea(cnt)
      if 50 < area_px <= max_area_px:
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
# MAIN APP LOGIC (JIKA FILE TERSEDIA)
# ============================================================

if file_bytes is not None:
  raw_polygons = extract_polygons_from_bytes(file_bytes)

  if not raw_polygons:
    st.warning(
        "⏳ No pattern detected yet. Make sure the component shape is drawn"
        " fully closed."
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
      # PERHITUNGAN STANDAR METRIK (CLEAN FORMATTING & YIELD & COST)
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

      # Perhitungan Biaya Material per Pasang (dalam USD)
      total_material_cost = used_length_m * material_price
      cost_per_pair = (
          total_material_cost / pairs_completed if pairs_completed > 0 else 0.0
      )

      st.markdown("### 📊 Summary Report")

      col_m1, col_m2, col_m3, col_m4, col_m5, col_m6, col_m7 = st.columns(7)
      col_m1.metric("Parts / pair", f"{2.00:.2f}")
      col_m2.metric("Net Area / pair", f"{net_area_per_pair:.4f}")
      col_m3.metric("Gross Area / pair", f"{gross_area_per_pair:.4f}")
      col_m4.metric("Waste / pair", f"{waste_area_per_pair:.4f}")
      col_m5.metric("Efficiency", f"{efficiency:.2f}%")
      col_m6.metric("Yield (m/pair)", f"{yield_value:.4f}")
      col_m7.metric("Material Cost / Pair", f"${cost_per_pair:.2f}")

      st.info(
          f"💡 **Production Info:** Placed {pairs_completed} pairs"
          f" ({pieces_completed} pcs) | Used Length: **{max_used_y:.2f} cm**"
          f" out of {sheet_length:.1f} cm (Width: {sheet_width:.1f} cm) | Total"
          f" Cost: **${total_material_cost:.2f}**"
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
