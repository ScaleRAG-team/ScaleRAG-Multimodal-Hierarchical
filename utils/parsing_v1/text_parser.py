import fitz
import re

def extract_text_sections(doc, doc_id):
    result = {"doc_id": doc_id, "title": "", "sections": []}
    font_counts = {}

    # Step 1: Find dominant font size (body text)
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") == 0:
                for line in b["lines"]:
                    for span in line["spans"]:
                        fid = (span["font"], round(span["size"], 1))
                        font_counts[fid] = font_counts.get(fid, 0) + len(span["text"])
    if not font_counts:
        raise ValueError("No text found")

    _, body_font_size = max(font_counts, key=font_counts.get)

    # Step 2: Extract title from largest text on first page
    first_page = doc[0]
    max_size = 0
    for b in first_page.get_text("dict")["blocks"]:
        if b.get("type") == 0:
            for line in b["lines"]:
                for span in line["spans"]:
                    if span["size"] > max_size:
                        max_size = span["size"]
                        result["title"] = " ".join([span["text"] for span in line["spans"]]).strip()

    current_section = None

    def start_new_section(title):
        nonlocal current_section
        if current_section:
            result["sections"].append(current_section)
        current_section = {"section_title": title.strip(), "blocks": []}

    heading_keywords = {
        "abstract", "references", "conclusion", "conclusions",
        "acknowledgments", "acknowledgements", "contents", "appendix"
    }

    # Process each page, preserving column order
    for page_num, page in enumerate(doc, start=1):
        # Get all text blocks on the page
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        page_width = page.rect.width
        # Determine if page has multiple columns by checking x positions
        left_blocks = []
        right_blocks = []
        for b in blocks:
            x0 = b["bbox"][0]
            # Heuristic: use half page width as column divider
            if x0 >= page_width * 0.5:
                right_blocks.append(b)
            else:
                left_blocks.append(b)
        # Sort blocks within each column top-to-bottom
        left_blocks.sort(key=lambda b: b["bbox"][1])
        right_blocks.sort(key=lambda b: b["bbox"][1])
        ordered_blocks = left_blocks + right_blocks

        # Merge nearby blocks into paragraphs (within the same column) based on vertical gap
        merged_paragraphs = []
        current_paragraph = []
        last_y = None

        for b in ordered_blocks:
            # Extract full text of the block, merging lines and handling hyphenation
            lines = ["".join([span["text"] for span in line["spans"]]).strip() for line in b["lines"]]
            # Merge hyphenated line breaks
            merged_lines = []
            i = 0
            while i < len(lines):
                if i < len(lines) - 1 and lines[i].endswith("-") and lines[i+1] and lines[i+1][0].islower():
                    # Merge current line (without trailing hyphen) with next line
                    merged_line = lines[i][:-1] + lines[i+1]
                    i += 1
                    # Continue merging if next line is also hyphenated
                    while i < len(lines) - 1 and merged_line.endswith("-") and lines[i+1] and lines[i+1][0].islower():
                        merged_line = merged_line[:-1] + lines[i+1]
                        i += 1
                    merged_lines.append(merged_line)
                else:
                    merged_lines.append(lines[i])
                i += 1
            block_text = " ".join(" ".join(merged_lines).split()).strip()
            # Normalize common ligatures and special characters
            block_text = block_text.replace("ﬁ", "fi").replace("ﬂ", "fl")

            if not block_text:
                continue

            # Filter out noise: page numbers, stray short text, etc.
            # Drop isolated page numbers in headers/footers
            if block_text.isdigit() and len(block_text) <= 2:
                if b["bbox"][1] < page.rect.height * 0.1 or b["bbox"][1] > page.rect.height * 0.9:
                    continue
            # Drop very short letter fragments
            if len(block_text) <= 4 and block_text[0].isalpha() and block_text[0].isupper() and block_text[1:].islower():
                continue
            # Drop standalone punctuation or bullets
            if len(block_text) <= 2 and re.match(r'^[\\W_]+$', block_text):
                continue

            y_top = b["bbox"][1]
            if last_y is not None and abs(y_top - last_y) > 15:
                merged_paragraphs.append(current_paragraph)
                current_paragraph = []
            current_paragraph.append((block_text, b))
            last_y = b["bbox"][3]

        if current_paragraph:
            merged_paragraphs.append(current_paragraph)

        for paragraph in merged_paragraphs:
            para_text = " ".join(t for t, _ in paragraph)
            para_text = " ".join(para_text.split())  # clean excess whitespace
            # Use first block's properties for the merged paragraph
            first_block = paragraph[0][1]

            # Heuristic to identify section headings
            is_heading = (
                len(para_text.split()) <= 12 and
                len(para_text) <= 120 and
                not para_text.strip().isdigit() and
                (
                    any(
                        span["size"] > body_font_size + 0.5 or
                        ("Bold" in span["font"] and span["size"] >= body_font_size)
                        for line in first_block["lines"] for span in line["spans"]
                    ) or
                    para_text.lower() in heading_keywords or
                    re.match(r"^(\d+[\.\d]*\s+)?[A-Z]", para_text)
                )
            )

            if is_heading:
                start_new_section(para_text)
                continue

            if current_section is None:
                current_section = {"section_title": "Introduction", "blocks": []}

            current_section["blocks"].append({
                "type": "text",
                "text": para_text,
                "page": page_num,
                "bbox": [round(x, 2) for x in first_block["bbox"]]
            })

    if current_section:
        result["sections"].append(current_section)

    return result
