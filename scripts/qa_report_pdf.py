"""Rasterize the final report and create contact sheets for visual QA."""

from pathlib import Path

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output" / "pdf" / "CIS6005_WRIT1_MEDA_Report.pdf"
OUT = ROOT / "tmp" / "report_render"
OUT.mkdir(parents=True, exist_ok=True)

doc = fitz.open(PDF)
page_paths = []
for index, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
    path = OUT / f"page-{index + 1:02d}.png"
    pix.save(path)
    page_paths.append(path)

thumb_w = 306
thumb_h = 396
label_h = 24
for sheet_index in range(0, len(page_paths), 6):
    batch = page_paths[sheet_index : sheet_index + 6]
    sheet = Image.new("RGB", (thumb_w * 3, (thumb_h + label_h) * 2), "#d7dce2")
    draw = ImageDraw.Draw(sheet)
    for local_index, path in enumerate(batch):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w - 12, thumb_h - 12))
        x = (local_index % 3) * thumb_w + (thumb_w - image.width) // 2
        y = (local_index // 3) * (thumb_h + label_h) + label_h
        sheet.paste(image, (x, y))
        draw.text((x, y - 20), f"Page {sheet_index + local_index + 1}", fill="#111827")
    sheet.save(OUT / f"contact-{sheet_index // 6 + 1}.png")

print(f"pages={len(page_paths)}")
print(f"size_points={doc[0].rect.width:.1f}x{doc[0].rect.height:.1f}")
print(f"contacts={(len(page_paths) + 5) // 6}")
