from pathlib import Path
import fitz 
import re

def _export_page_images(doc, page, img_dir: Path, stem: str, pno: int):
    """Return {xref: image_path_or_None}. Exports unique bitmap XObjects."""
    paths = {}
    seen = set()
    for info in page.get_images(full=True):  # (xref, smask, width, height, bpc, colorspace, alt, name, ...)
        xref = info[0]
        smask = info[1]
        if xref in seen or smask > 0:  # skip soft masks; they’re not content images
            continue
        seen.add(xref)
        try:
            base = doc.extract_image(xref)
            ext = base.get("ext", "png")
            out_path = img_dir / f"{stem}_p{pno}_x{xref}.{ext}"
            with open(out_path, "wb") as f:
                f.write(base["image"])
            paths[xref] = str(out_path)
        except Exception:
            paths[xref] = None
    return paths



def _safe_rect(page, bbox, pad=0.5):
    # intersect with page; add a tiny pad to avoid zero-area after rounding
    r = fitz.Rect(bbox)
    r = r & page.rect
    if r.is_empty or r.width <= 0 or r.height <= 0:
        return None
    r.x0 = max(page.rect.x0, r.x0 - pad)
    r.y0 = max(page.rect.y0, r.y0 - pad)
    r.x1 = min(page.rect.x1, r.x1 + pad)
    r.y1 = min(page.rect.y1, r.y1 + pad)
    return r




def _raster_crop(page, rect: fitz.Rect, dpi: int, out_path: Path):
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    pix.save(str(out_path))











def _areas_from_caption_bbox(cbx, page_w, page_h, pad=6):
    """Return FOUR candidate areas for Camelot (strings x1,y1,x2,y2):
       above/below × two Y-origin conversions (robust)."""

    
    x0, y0, x1, y1 = cbx  # PyMuPDF coords (y down)
    # Regions in PyMuPDF coords
    above = (0+pad, 0+pad, page_w-pad, max(0, y0-pad))
    below = (0+pad, min(page_h, y1+pad), page_w-pad, page_h-pad)

    def to_camelot(a):
        # convert PyMuPDF rect -> Camelot (origin bottom-left)
        ax0, ay0, ax1, ay1 = a
        # conversion A (correct): y' = page_h - y
        ca = f"{ax0},{page_h-ay1},{ax1},{page_h-ay0}"
        # conversion B (fallback): no flip (handles odd producer PDFs)
        cb = f"{ax0},{ay0},{ax1},{ay1}"
        return ca, cb

    A1, B1 = to_camelot(above)
    A2, B2 = to_camelot(below)
    return [A1, B1, A2, B2]  # try in this order



