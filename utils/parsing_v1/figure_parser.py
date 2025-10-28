import re
import fitz  # PyMuPDF
import os

def extract_figures_from_pdf(pdf_path, output_dir="output", doc_id=None, result=None):
    doc = fitz.open(pdf_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if doc_id is None:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    figure_blocks = []
    figure_count = 0

    for page_number, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        for b in blocks:
            if b.get("type") != 0:
                continue

            lines_text = ["".join([span["text"] for span in line["spans"]]).strip() for line in b["lines"]]
            block_text = " ".join(lines_text)
            caption_text = " ".join(block_text.split())

            # Detect figure caption
            if not re.match(r'^(Figure\s*\d+\.|Fig\.?\s*\d+[:\.])', caption_text, re.IGNORECASE):
                continue

            figure_count += 1
            x0, y0, x1, y1 = b["bbox"]
            page_width = page.rect.width
            # Small horizontal padding around image
            region_x0 = max(0, x0 - 5)
            region_x1 = min(page_width, x1 + 5)

            # Estimate figure area above the caption
            prev_block_bottom = None
            for pb in blocks:
                if pb is b:
                    break
                if pb.get("type") == 0:
                    prev_block_bottom = pb["bbox"][3]
            if prev_block_bottom and prev_block_bottom < y0 - 20:
                top_bound = prev_block_bottom
            else:
                top_bound = max(0, y0 - (page.rect.height * 0.5))
            region_y0 = top_bound
            region_y1 = y0  # top of caption

            # Extract figure image region
            clip_rect = fitz.Rect(region_x0, region_y0, region_x1, region_y1)
            pix = page.get_pixmap(clip=clip_rect, dpi=150)
            image_filename = f"{doc_id}_figure_{figure_count}.png"
            image_path = os.path.join(output_dir, image_filename)
            pix.save(image_path)

            figure_block = {
                "type": "figure",
                "caption": caption_text,
                "page": page_number,
                "image_path": image_path
            }
            # Include figure identifier filename
            figure_block["figure_id"] = image_filename

            figure_blocks.append(figure_block)

            # Insert figure block into result JSON in place of the caption text
            if result:
                inserted = False
                for sec in result["sections"]:
                    for i, blk in enumerate(sec["blocks"]):
                        if blk["type"] == "text" and blk["text"].strip() == caption_text.strip():
                            sec["blocks"][i] = figure_block
                            inserted = True
                            break
                    if inserted:
                        break
                if not inserted:
                    # Fallback: if caption not found, append figure to a separate section
                    if not result["sections"]:
                        result["sections"].append({"section_title": "Figures", "blocks": []})
                    result["sections"][-1]["blocks"].append(figure_block)
    doc.close()
    return figure_blocks
