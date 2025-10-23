import json, fitz, io
from PIL import Image
from pathlib import Path

def render_bbox_from_json(json_path: str, pdf_path: str, idx: int = 0, dpi: int = 200):
    """
    Renders an image region from a parsed PyMuPDF JSON using its bbox + page number.
    idx = index of the picture item to display.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    pics = data.get("pictures", [])
    if not pics:
        raise ValueError("No pictures found in JSON.")
    if idx >= len(pics):
        raise IndexError(f"Index {idx} out of range. Only {len(pics)} pictures available.")

    pic = pics[idx]
    bbox = pic.get("bbox")
    page_no = pic.get("page_no", 1)

    if not bbox:
        raise ValueError("No bbox found for this picture.")

    with fitz.open(pdf_path) as doc:
        page = doc[page_no - 1]
        rect = fitz.Rect(*bbox)
        pix = page.get_pixmap(clip=rect, dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img