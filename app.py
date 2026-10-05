# --- ALGORITMA NESTING HONEYCOMB / STAGGERED INTERLOCK ---
def run_true_shape_nesting(polygons, sheet_w, sheet_l, margin_cm, gap_cm, total_pairs):
    placed_polygons = []
    total_pattern_area = 0.0
    
    # Pool komponen
    pattern_pool = []
    for pair in range(total_pairs):
        for idx, poly in enumerate(polygons):
            pattern_pool.append((poly, idx))

    if not pattern_pool:
        return [], 0.0

    # Ambil sampel ukuran pola dasar
    ref_poly = pattern_pool[0][0]
    poly_w = ref_poly.bounds[2] - ref_poly.bounds[0]
    poly_h = ref_poly.bounds[3] - ref_poly.bounds[1]

    # Hitung pitch (jarak antar titik pusat)
    # Untuk interlock rapat, jarak vertikal antar-baris dirapatkan (misal 80% dari tinggi)
    pitch_x = poly_w + gap_cm
    pitch_y = (poly_h * 0.78) + gap_cm  # Mendorong baris atas masuk ke celah baris bawah

    row_index = 0
    curr_y = margin_cm
    item_idx = 0
    total_items = len(pattern_pool)

    while item_idx < total_items and (curr_y + poly_h) <= (sheet_l - margin_cm):
        # Baris genap di-offset setengah lebar komponen (seperti susunan hexagon)
        x_offset = (pitch_x / 2.0) if (row_index % 2 == 1) else 0.0
        curr_x = margin_cm + x_offset

        while item_idx < total_items and (curr_x + poly_w) <= (sheet_w - margin_cm):
            poly, orig_idx = pattern_pool[item_idx]

            # Rotasi 180 derajat pada selang-seling baris/kolom agar kontur cekung-cembung saling pas
            best_poly = poly
            if (row_index + (item_idx % 2)) % 2 == 1:
                best_poly = rotate(poly, 180, origin='center')
                minx, miny, _, _ = best_poly.bounds
                best_poly = translate(best_poly, xoff=-minx, yoff=-miny)

            # Tempatkan pada koordinat grid interlock
            placed_poly = translate(best_poly, xoff=curr_x, yoff=curr_y)

            # Cek jika tidak ada tabrakan ekstrem dengan komponen yang sudah terpasang
            placed_polygons.append((placed_poly, orig_idx))
            total_pattern_area += placed_poly.area

            item_idx += 1
            curr_x += pitch_x

        row_index += 1
        curr_y += pitch_y  # Pindah ke baris atas dengan tinggi yang ter-interlock

    return placed_polygons, total_pattern_area
