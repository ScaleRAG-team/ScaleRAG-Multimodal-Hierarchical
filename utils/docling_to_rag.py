# utils/docling_to_rag.py

import os
import re
import json
from pathlib import Path
from tqdm import tqdm


def convert_docling_to_rag(doc, paper_id):
    """
    Convert a Docling JSON object into RAG-format blocks.
    Each block has: id, type, content, metadata.
    """
    rag_blocks = []
    # State variables
    current_section = None
    had_title = False  # flag to indicate if we've passed the paper title
    
    # Counters for unique IDs
    para_count = fig_count = table_count = eq_count = 1
    
    # Shortcut references to lists in the doc
    texts = doc.get("texts", [])
    pictures = doc.get("pictures", [])
    tables = doc.get("tables", [])
    groups = doc.get("groups", [])
    
    def add_block(block_type, content, page_no, bbox):
        """Helper to add a content block to the output list with proper schema."""
        nonlocal para_count, fig_count, table_count, eq_count
        
        # Clean and normalize content text
        content_str = content.replace("\n", " ")
        content_str = re.sub(r'GLYPH<\d+>', '', content_str)  # remove placeholder tokens if any
        content_str = content_str.strip()
        if content_str == "":
            return  # skip empty content
        # Skip blocks that are just numbers or single characters (page numbers, stray markers)
        if len(content_str) <= 1 or content_str.isdigit() or re.match(r'^\(?\d+\)?$', content_str):
            return
        
        # Determine block ID based on type and increment the corresponding counter
        if block_type == "paragraph":
            block_id = f"{paper_id}_paragraph_{para_count}"
            para_count += 1
        elif block_type == "figure":
            block_id = f"{paper_id}_fig_{fig_count}"
            fig_count += 1
        elif block_type == "table":
            block_id = f"{paper_id}_table_{table_count}"
            table_count += 1
        elif block_type == "equation":
            block_id = f"{paper_id}_equation_{eq_count}"
            eq_count += 1
        else:
            # Fallback for unexpected types
            block_id = f"{paper_id}_{block_type}_{para_count}"
            para_count += 1
        
        # Build metadata
        meta = {}
        if page_no is not None:
            meta["page"] = page_no
        if current_section:
            meta["section"] = current_section
        if bbox:
            # Use coordinates if available (left, bottom, right, top from Docling prov)
            x0, y0 = bbox.get("l"), bbox.get("b")
            x1, y1 = bbox.get("r"), bbox.get("t")
            meta["bbox"] = [x0, y0, x1, y1]
        
        # Append the block to output list
        rag_blocks.append({
            "id": block_id,
            "type": block_type,
            "content": content_str,
            "metadata": meta
        })
    
    def process_children(children):
        """Recursively process a list of child references (from body or group)."""
        nonlocal current_section, had_title
        for ref in children:
            ref_str = ref.get("$ref", "")
            if not ref_str.startswith("#/"):
                continue  # skip if not a valid reference
            # Parse reference like "#/texts/10" into category and index
            try:
                cat, idx = ref_str[2:].split('/')
                idx = int(idx)
            except ValueError:
                continue
            # Handle each category of reference
            if cat == "texts":
                if idx < 0 or idx >= len(texts):
                    continue
                item = texts[idx]
                label = item.get("label", "")
                layer = item.get("content_layer", "")
                text_val = item.get("text", "") or ""
                orig_val = item.get("orig", "") or ""
                # Determine page number and bbox if available
                page_no = None
                bbox = None
                if item.get("prov"):
                    page_no = item["prov"][0].get("page_no")
                    bbox = item["prov"][0].get("bbox")
                
                # Skip non-body content (headers/footers, etc.)
                if layer != "body":
                    continue
                
                if label == "section_header":
                    # Update current section context (skip outputting the header itself)
                    sec_text = text_val.strip() or orig_val.strip()
                    # Remove leading numbers or numbering (e.g., "1 Introduction" -> "Introduction")
                    sec_text_clean = re.sub(r'^[0-9.\s]+', '', sec_text).strip()
                    if not had_title:
                        # If this is the first section header encountered, check if it's the paper title
                        if sec_text_clean.lower() not in ["abstract", "introduction", "related work", "background", 
                                                          "conclusion", "acknowledgments", "acknowledgements", 
                                                          "references", "appendix"] and not re.match(r'^\d', sec_text):
                            # Treat it as the title and skip setting section (no output)
                            had_title = True
                            continue
                        # Otherwise, it's a real section (like "Abstract" or a numbered section)
                        had_title = True
                        current_section = sec_text_clean
                    else:
                        # Subsequent section headers
                        current_section = sec_text_clean
                    continue  # don't output section headers as blocks
                
                if label in ["page_header", "page_footer"]:
                    # Skip page headers/footers and other furniture text
                    continue
                
                # For regular text content:
                if current_section is None:
                    # We haven't hit a section heading yet – likely front matter (authors, etc.)
                    combined = text_val.strip() or orig_val.strip()
                    if len(combined) > 100 and re.search(r'[.!?]\s*$', combined):
                        # If this looks like a long paragraph ending in punctuation, assume it's the Abstract
                        current_section = "Abstract"
                        had_title = True
                        add_block("paragraph", combined, page_no, bbox)
                    else:
                        # Otherwise, skip (e.g., author names, affiliations, or other metadata)
                        continue
                else:
                    # We have a section context set, so this is content within a section
                    if text_val.strip() == "":
                        # If the recognized text is empty but orig has content, treat as equation block
                        if orig_val.strip():
                            # Use original text for equation (after removing any placeholders)
                            eq_text = re.sub(r'GLYPH<\d+>', '', orig_val.strip())
                            add_block("equation", eq_text, page_no, bbox)
                        # if orig_val is also empty, skip (nothing to add)
                    else:
                        # There is textual content
                        content_str = text_val
                        # Heuristic: If the text contains mathematical symbols and few normal words, classify as equation
                        if re.search(r'=|≥|≤|≠|∑|∫|√|∈|⊥|⊤|∀|∃|→|←', content_str):
                            # Count alphabetic word tokens (length >=4 as a proxy for normal text)
                            word_tokens = re.findall(r'[A-Za-z]{4,}', content_str)
                            if len(word_tokens) < 2:
                                add_block("equation", content_str, page_no, bbox)
                                continue
                        # Otherwise, treat it as a normal paragraph
                        add_block("paragraph", content_str, page_no, bbox)
            
            elif cat == "pictures":
                # Handle figure (picture) reference
                if idx < 0 or idx >= len(pictures):
                    continue
                pic = pictures[idx]
                page_no = None
                bbox = None
                if pic.get("prov"):
                    page_no = pic["prov"][0].get("page_no")
                    bbox = pic["prov"][0].get("bbox")
                # Gather caption text from associated text items
                caption_texts = []
                if pic.get("captions"):
                    # If captions list is present, collect text from those references
                    for cap_ref in pic["captions"]:
                        cap_ref_str = cap_ref.get("$ref", "")
                        if cap_ref_str.startswith("#/texts/"):
                            try:
                                cap_idx = int(cap_ref_str.split('/')[2])
                            except:
                                continue
                            if 0 <= cap_idx < len(texts):
                                cap_item = texts[cap_idx]
                                cap_content = cap_item.get("text", "") or cap_item.get("orig", "") or ""
                                caption_texts.append(cap_content)
                else:
                    # If no explicit captions list, search children for a caption
                    for child_ref in pic.get("children", []):
                        cref_str = child_ref.get("$ref", "")
                        if cref_str.startswith("#/texts/"):
                            try:
                                cap_idx = int(cref_str.split('/')[2])
                            except:
                                continue
                            if 0 <= cap_idx < len(texts):
                                cap_item = texts[cap_idx]
                                # Identify caption by label or text starting with "Figure"
                                if cap_item.get("label") == "caption" or cap_item.get("text", "").strip().lower().startswith("figure"):
                                    cap_content = cap_item.get("text", "") or cap_item.get("orig", "") or ""
                                    caption_texts.append(cap_content)
                if caption_texts:
                    caption_full = " ".join([c.strip() for c in caption_texts])
                    add_block("figure", caption_full, page_no, bbox)
                # If no caption found, skip output for this figure
            
            elif cat == "tables":
                # Handle table reference
                if idx < 0 or idx >= len(tables):
                    continue
                tbl = tables[idx]
                page_no = None
                bbox = None
                if tbl.get("prov"):
                    page_no = tbl["prov"][0].get("page_no")
                    bbox = tbl["prov"][0].get("bbox")
                caption_texts = []
                if tbl.get("captions"):
                    for cap_ref in tbl["captions"]:
                        cap_ref_str = cap_ref.get("$ref", "")
                        if cap_ref_str.startswith("#/texts/"):
                            try:
                                cap_idx = int(cap_ref_str.split('/')[2])
                            except:
                                continue
                            if 0 <= cap_idx < len(texts):
                                cap_item = texts[cap_idx]
                                cap_content = cap_item.get("text", "") or cap_item.get("orig", "") or ""
                                caption_texts.append(cap_content)
                else:
                    # Search children for caption if not in captions list
                    for child_ref in tbl.get("children", []):
                        cref_str = child_ref.get("$ref", "")
                        if cref_str.startswith("#/texts/"):
                            try:
                                cap_idx = int(cref_str.split('/')[2])
                            except:
                                continue
                            if 0 <= cap_idx < len(texts):
                                cap_item = texts[cap_idx]
                                if cap_item.get("label") == "caption" or cap_item.get("text", "").strip().lower().startswith("table"):
                                    cap_content = cap_item.get("text", "") or cap_item.get("orig", "") or ""
                                    caption_texts.append(cap_content)
                if caption_texts:
                    caption_full = " ".join([c.strip() for c in caption_texts])
                    add_block("table", caption_full, page_no, bbox)
                # If no caption text, skip this table
            
            elif cat == "groups":
                # Handle a group (e.g., a list of items)
                if idx < 0 or idx >= len(groups):
                    continue
                grp = groups[idx]
                # Recursively process the children of this group
                process_children(grp.get("children", []))
            # (We ignore other categories like "figures" or "equations" if not present, as well as footnotes if any)
    
    # Start processing from the top-level body children
    process_children(doc.get("body", {}).get("children", []))
    return rag_blocks


def batch_convert_docling_to_rag(input_dir="data/docling_json", output_dir="data/rag_json"):
    """Convert all Docling JSON files in input_dir to RAG JSON format."""
    os.makedirs(output_dir, exist_ok=True)
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    print(f"Found {len(input_files)} Docling JSON files in '{input_dir}'.")

    for filename in tqdm(input_files, desc="Converting Docling → RAG", unit="file"):
        input_path = os.path.join(input_dir, filename)
        paper_id = re.sub(r"\.docling\.json$", "", filename)
        with open(input_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        rag_blocks = convert_docling_to_rag(doc, paper_id)

        output_path = os.path.join(output_dir, f"{paper_id}.rag.json")
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(rag_blocks, f_out, ensure_ascii=False, indent=2)

        print(f"Converted {filename} → {paper_id}.rag.json ({len(rag_blocks)} blocks)")
    
    return {
    "total_files": len(input_files),
    "output_dir": output_dir}
