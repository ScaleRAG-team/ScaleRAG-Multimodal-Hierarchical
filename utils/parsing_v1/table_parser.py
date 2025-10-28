import os
import re
import camelot
import pdfplumber
import pandas as pd

def extract_tables(doc, doc_id, output_dir, pdf_path, result):
    table_count = 0
    page_count = doc.page_count

    for page_num, page in enumerate(doc, start=1):
        page_height = page.rect.height
        page_width = page.rect.width
        blocks = page.get_text("dict")["blocks"]
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        for b in blocks:
            if b.get("type") != 0:
                continue

            # Extract potential caption text for this block
            lines_text = ["".join([span["text"] for span in line["spans"]]).strip() for line in b["lines"]]
            caption_text = " ".join(" ".join(lines_text).split())
            # Look for "Table X." or "Table X:" at start
            if not re.match(r"^(Table\s*\d+(\.|:))", caption_text, re.IGNORECASE):
                continue

            # We've identified a table caption
            table_count += 1

            # Determine the region below the caption to search for the table
            x0, y0, x1, y1 = b["bbox"]
            region_top = y1  # start just below caption bottom
            region_bottom = min(page_height, y1 + 250)  # search up to 250 points below caption
            region_x0 = max(0, x0 - 10)
            region_x1 = min(page_width, x1 + 10)

            region_str = f"{region_x0},{page_height - region_top},{region_x1},{page_height - region_bottom}"

            # Try extracting table using Camelot
            tables = []
            try:
                tables = camelot.read_pdf(pdf_path, flavor="stream", pages=str(page_num), table_regions=[region_str], strip_text="\n")
            except Exception as e:
                print(f"[Warning] Camelot (stream) failed on page {page_num}: {e}")
            # If Camelot stream found nothing, try lattice mode
            if not tables or len(tables) == 0 or tables[0].df.shape[0] == 0:
                try:
                    tables = camelot.read_pdf(pdf_path, flavor="lattice", pages=str(page_num), table_regions=[region_str])
                except Exception as e:
                    print(f"[Warning] Camelot (lattice) failed on page {page_num}: {e}")

            csv_path = ""
            image_path = ""
            table_extracted = False
            if tables and len(tables) > 0 and tables[0].df.shape[0] > 0:
                # Save the first table found to CSV
                csv_path = os.path.join(output_dir, f"{doc_id}_table_{table_count}.csv")
                tables[0].df.to_csv(csv_path, index=False)
                table_extracted = True
            else:
                # Camelot failed or returned empty; try pdfplumber
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        pdf_page = pdf.pages[page_num - 1]
                        # Crop to the same region defined for Camelot
                        cropped = pdf_page.crop((region_x0, region_top, region_x1, region_bottom))
                        table_data = cropped.extract_table(table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"})
                        if table_data is None:
                            table_data = cropped.extract_table(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
                    if table_data:
                        # Save table_data (list of lists) to CSV using pandas
                        df = pd.DataFrame(table_data)
                        csv_path = os.path.join(output_dir, f"{doc_id}_table_{table_count}.csv")
                        # Do not include header if not present
                        df.to_csv(csv_path, index=False, header=False)
                        table_extracted = True
                except Exception as e:
                    print(f"[Warning] pdfplumber failed on page {page_num}: {e}")

            if not table_extracted:
                # Final fallback: render table region as image for manual review/OCR
                pix = page.get_pixmap(clip=fitz.Rect(region_x0, region_top, region_x1, region_bottom), dpi=150)
                image_path = os.path.join(output_dir, f"{doc_id}_table_{table_count}.png")
                pix.save(image_path)
                print(f"[Info] Table on page {page_num} saved as image (fallback): {image_path}")

            # Create table block metadata
            table_block = {
                "type": "table",
                "caption": caption_text,
                "page": page_num,
                "csv_path": csv_path
            }
            if image_path:
                table_block["image_path"] = image_path

            # Insert the table block into the result in reading order by replacing the caption text block
            caption_found = False
            if result:
                for sec in result["sections"]:
                    for i, blk in enumerate(sec["blocks"]):
                        if blk["type"] == "text" and blk["text"].strip() == caption_text.strip():
                            sec["blocks"][i] = table_block
                            caption_found = True
                            break
                    if caption_found:
                        break
            if not caption_found:
                # If not found (e.g., caption text was merged or not in text), append at end of result
                if not result["sections"]:
                    result["sections"].append({"section_title": "Tables", "blocks": []})
                result["sections"][-1]["blocks"].append(table_block)
