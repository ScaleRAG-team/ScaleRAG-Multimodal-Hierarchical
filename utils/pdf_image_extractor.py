# utils/pdf_image_extractor.py

import os
import re
import json
import fitz
import difflib
from pathlib import Path
from tqdm import tqdm


# Utility Functions

def docling_bbox_to_rect(bbox, page):
    """Convert Docling bbox (bottom-left origin) to PyMuPDF Rect (top-left origin)."""
    h = page.rect.height
    return fitz.Rect(bbox["l"], h - bbox["t"], bbox["r"], h - bbox["b"])


def crop_region(pdf_page, bbox, out_path, dpi=200):
    """Crop a region from a PDF page and save as PNG."""
    rect = docling_bbox_to_rect(bbox, pdf_page)
    pix = pdf_page.get_pixmap(clip=rect, dpi=dpi)
    pix.save(str(out_path))


def norm_txt(s):
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def best_match(target, candidates, cutoff=0.6):
    """Return index of best matching candidate caption by similarity ratio."""
    if not candidates:
        return None
    ratios = [
        difflib.SequenceMatcher(None, norm_txt(target), norm_txt(c)).ratio()
        for c in candidates
    ]
    i = max(range(len(ratios)), key=lambda k: ratios[k])
    return i if ratios[i] >= cutoff else None


# Main Function

def enrich_rag_with_images(
    pdf_dir="data2/pdf",
    docling_dir="data2/docling_json",
    rag_dir="data2/rag_json",
    assets_dir="data2/rag_assets",
    cutoff=0.55,
):
    """Attach cropped figure/table images to RAG JSONs and save enriched outputs."""
    pdf_dir = Path(pdf_dir)
    docling_dir = Path(docling_dir)
    rag_dir = Path(rag_dir)
    assets_dir = Path(assets_dir)
    (assets_dir / "images").mkdir(parents=True, exist_ok=True)

    updated = 0
    rag_paths = sorted(rag_dir.glob("*.rag.json"))

    for rag_path in tqdm(rag_paths, desc="Enriching RAG JSONs with images"):
        stem = rag_path.stem.replace(".rag", "")
        pdf_path = pdf_dir / f"{stem}.pdf"
        docling_path = docling_dir / f"{stem}.docling.json"

        if not (pdf_path.exists() and docling_path.exists()):
            print(f"Skip {stem}: missing PDF or Docling JSON")
            continue

        # Load all related files
        rag_blocks = json.loads(rag_path.read_text(encoding="utf-8"))
        docling = json.loads(docling_path.read_text(encoding="utf-8"))

        texts = docling.get("texts", [])
        pictures = docling.get("pictures", [])
        tables = docling.get("tables", [])

        # Caption extraction helper
        def cap_text_list(obj):
            caps = []
            if obj.get("captions"):
                for ref in obj["captions"]:
                    if ref.get("$ref", "").startswith("#/texts/"):
                        idx = int(ref["$ref"].split("/")[-1])
                        t = texts[idx].get("text") or texts[idx].get("orig") or ""
                        caps.append(t.strip())
            else:
                for ref in obj.get("children", []):
                    if ref.get("$ref", "").startswith("#/texts/"):
                        idx = int(ref["$ref"].split("/")[-1])
                        lab = texts[idx].get("label")
                        t = texts[idx].get("text") or texts[idx].get("orig") or ""
                        if lab == "caption" or t.strip().lower().startswith(("figure", "table")):
                            caps.append(t.strip())
            return caps

        pic_caps = [" ".join(cap_text_list(p)) for p in pictures]
        tbl_caps = [" ".join(cap_text_list(t)) for t in tables]

        pdf = fitz.open(str(pdf_path))
        changed = False

        for b in rag_blocks:
            if b.get("type") not in ("figure", "table"):
                continue

            caption = b.get("content", "").strip()
            if not caption:
                continue

            objs = pictures if b["type"] == "figure" else tables
            caps = pic_caps if b["type"] == "figure" else tbl_caps
            subdir = "images"

            j = best_match(caption, caps, cutoff=cutoff)
            if j is None:
                continue

            obj = objs[j]
            prov = (obj.get("prov") or [None])[0]
            if not prov or not prov.get("bbox") or not prov.get("page_no"):
                continue

            try:
                page_no = prov["page_no"] - 1
                bbox = prov["bbox"]
                page = pdf[page_no]

                paper_dir = assets_dir / subdir / stem / b["type"]
                paper_dir.mkdir(parents=True, exist_ok=True)

                out_name = f"{b['type']}_{j+1}.png"
                out_path = paper_dir / out_name
                crop_region(page, bbox, out_path, dpi=220)

                md = b.setdefault("metadata", {})
                md["page"] = md.get("page", prov.get("page_no"))
                md["bbox"] = md.get("bbox", [bbox["l"], bbox["b"], bbox["r"], bbox["t"]])
                md["image_path"] = str(
                    Path("data") / "rag_assets" / "images" / stem / b["type"] / out_name
                )

                changed = True
            except Exception:
                continue

        pdf.close()
        if changed:
            rag_path.write_text(
                json.dumps(rag_blocks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            updated += 1
            print(f"Enriched {rag_path.name}")

    print(f"Done. Updated {updated} RAG JSON files with image paths.")
    return {"updated_files": updated, "rag_dir": str(rag_dir)}
