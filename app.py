# BENTUK PREVIEW GRID 2 BARIS (DIPERBAIKI)
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

    else:  # Category 4: Two Way Staggered (Diperbaiki agar selang-seling P1 dan P2)
      default_stagger_x = step_x / 2
      preview_items.append((p1, 0))
      preview_items.append((translate(p2, xoff=step_x, yoff=0), 1))
      row2_y = pitch_y
      row2_x1 = default_stagger_x + preview_shift_x
      row2_x2 = row2_x1 + step_x
      preview_items.append(
          (translate(p1, xoff=row2_x1, yoff=row2_y), 0)
      )  # Kolom 1 Baris 2 (P1)
      preview_items.append(
          (translate(p2, xoff=row2_x2, yoff=row2_y), 1)
      )  # Kolom 2 Baris 2 (P2 / Terbalik)
