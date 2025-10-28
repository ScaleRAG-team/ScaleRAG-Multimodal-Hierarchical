import os
import re
import fitz

def extract_equations(doc, doc_id, output_dir, result):
    eq_count = 0
    last_eq_block = None

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get("type") != 0:
                continue
            # Combine all spans in the line
            lines = ["".join([span["text"] for span in line["spans"]]) for line in b["lines"]]
            if len(lines) != 1:
                continue  # Only consider single-line blocks
            line_text = lines[0].strip()
            if not line_text:
                continue

            # Check if this line is just an equation number in parentheses
            if re.fullmatch(r"\(\d+\)", line_text):
                # If a standalone number block follows an equation, attach it
                if last_eq_block is not None:
                    num = int(line_text.strip("()"))
                    last_eq_block["number"] = num
                # Remove this number-only text block from result (if present)
                if result:
                    for sec in result["sections"]:
                        sec["blocks"] = [blk for blk in sec["blocks"] if not (blk["type"] == "text" and blk["text"].strip() == line_text)]
                # Skip creating a new block for the number alone
                continue

            # Identify displayed equation lines (either contain equation symbols or end with a number)
            if re.search(r"\(\d+\)$", line_text) or len(re.findall(r'[=+\-*/\\^]', line_text)) > 2 * len(re.findall(r'[A-Za-z]', line_text)):
                eq_count += 1
                # Determine if an equation number is embedded at end, and separate it
                eq_number = None
                m = re.search(r"\((\d+)\)$", line_text)
                if m:
                    eq_number = int(m.group(1))
                    line_content = line_text[:m.start()].strip()
                else:
                    line_content = line_text

                # Generate an image of the equation region
                clip = fitz.Rect(b["bbox"])
                image_path = os.path.join(output_dir, f"{doc_id}_equation_{eq_count}.png")
                page.get_pixmap(clip=clip, dpi=150).save(image_path)

                # Create equation block structure
                eq_block = {
                    "type": "equation",
                    "content": line_content,
                    "page": page_num,
                    "image_path": image_path
                }
                if eq_number is not None:
                    eq_block["number"] = eq_number

                # Insert the equation block at the proper place in result by replacing the original text block
                if result:
                    inserted = False
                    for sec in result["sections"]:
                        for i, blk in enumerate(sec["blocks"]):
                            if blk["type"] == "text" and blk["text"].strip() == line_text:
                                sec["blocks"][i] = eq_block
                                inserted = True
                                break
                        if inserted:
                            break
                    if not inserted:
                        # If not found (should be rare), append to last section
                        if not result["sections"]:
                            result["sections"].append({"section_title": "Equations", "blocks": []})
                        result["sections"][-1]["blocks"].append(eq_block)
                last_eq_block = eq_block
