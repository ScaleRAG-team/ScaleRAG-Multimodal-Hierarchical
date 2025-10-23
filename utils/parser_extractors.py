from typing import Callable, Dict, Any, List, Tuple
from pathlib import Path
import camelot
from collections.abc import Iterable 
import re

from parser_utils import _export_page_images, _safe_rect, _raster_crop, _areas_from_caption_bbox



# helper function for extract_text_blocks
# flattens text hierarchy in PDFs and concatenates all spans["text"] inside each line
def _merge_block_text(block: Dict[str, Any]) -> str:
    # PyMuPDF "dict" layout → concatenate spans
    lines = []
    for line in block.get("lines", []):

        # join lines and strip whitespaces
        pieces = [span.get("text", "") for span in line.get("spans", [])]
        if pieces:
            lines.append("".join(pieces))
    return "\n".join(lines).strip()



    

def extract_text_blocks(
    page,
    pno: int,
    merge: Callable[[Dict[str, Any]], str] = _merge_block_text,
    min_chars: int = 1,
) -> List[Dict[str, Any]]:
    """
    Return merged text blocks for a page.
    No side effects; caller decides how to store/serialize.
    """
    pdict = page.get_text("dict")
    out: List[Dict[str, Any]] = []
    for block in pdict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        bb: Tuple[float, float, float, float] = tuple(block.get("bbox", (0, 0, 0, 0)))
        txt = merge(block)
        if txt and len(txt) >= min_chars:
            out.append({"page_no": pno, "bbox": bb, "text": txt})
    return out




# required by extract_image_blocks function
_CAP_RE = re.compile(r"^(Figure|Fig\.?)\s*\d+[\.:]?", re.I)

def _w(bb): return max(0.0, bb[2] - bb[0])
def _h(bb): return max(0.0, bb[3] - bb[1])

def extract_image_blocks(
    doc,
    page,
    pdict: Dict[str, Any],
    *,
    pno: int,
    img_dir: Path,
    stem: str,
    raster_fallback: bool = True,
    raster_dpi: int = 220,
    min_img_area: float = 80*80,
    min_cap_overlap: float = 0.35,   # horiz overlap fraction of caption width
    max_v_gap: float = 60.0,         # max vertical gap between image bottom and caption top
    width_tol: float = 0.30,         # |img_w - cap_w| / cap_w ≤ width_tol
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Group page image blocks into FIGURES using caption anchoring.
    Returns:
      figures: [{page_no, bbox, caption, parts:[{bbox,xref,image_path}], image_path}]
      stats: summary counts
    """
    # 0) Export unique bitmaps on the page (XObjects)
    paths_by_xref = _export_page_images(doc, page, img_dir, stem, pno)
    xobjects_found = len(paths_by_xref)
    xobjects_exported = sum(1 for v in paths_by_xref.values() if v)

    # 1) Collect layout image blocks (from pdict)
    layout_imgs: List[Dict[str, Any]] = []
    for block in pdict.get("blocks", []):
        if block.get("type", 0) != 1:
            continue
        bb = tuple(block.get("bbox", (0, 0, 0, 0)))
        if _w(bb) * _h(bb) < min_img_area:
            continue
        xref = block.get("xref") or block.get("number")
        if not isinstance(xref, int):
            imginfo = block.get("image")
            if isinstance(imginfo, dict):
                xref = imginfo.get("xref")
        image_path = paths_by_xref.get(xref) if isinstance(xref, int) else None
        layout_imgs.append({"bbox": bb, "xref": xref, "image_path": image_path})

    layout_blocks = len(layout_imgs)

    # 2) Find caption blocks on this page
    captions = []
    for block in pdict.get("blocks", []):
        if block.get("type", 0) == 0:
            text = "".join(
                span.get("text", "")
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ).strip()
            if _CAP_RE.match(text):
                captions.append({"text": text, "bbox": tuple(block.get("bbox", (0,0,0,0)))})

    # 3) For each caption, group the image parts above it (same width band + overlap)
    assigned = set()  # indices of layout_imgs already grouped
    figures: List[Dict[str, Any]] = []
    for cap in captions:
        cx0, cy0, cx1, cy1 = cap["bbox"]
        cap_w = max(1e-6, _w(cap["bbox"]))
        cand_idxs: List[int] = []

        for idx, ib in enumerate(layout_imgs):
            if idx in assigned:
                continue
            bx0, by0, bx1, by1 = ib["bbox"]

            # must end above caption top
            if by1 > cy0:
                continue

            # sufficient horizontal overlap with caption width
            overlap = max(0.0, min(bx1, cx1) - max(bx0, cx0))
            if (overlap / cap_w) < min_cap_overlap:
                continue

            # vertical proximity
            dy = cy0 - by1
            if dy < 0 or dy > max_v_gap:
                continue

            # width similarity
            img_w = _w(ib["bbox"])
            if abs(img_w - cap_w) / cap_w > width_tol:
                continue

            cand_idxs.append(idx)

        if not cand_idxs:
            continue

        # union bbox of candidate parts
        ux0, uy0, ux1, uy1 = layout_imgs[cand_idxs[0]]["bbox"]
        for idx in cand_idxs[1:]:
            bx0, by0, bx1, by1 = layout_imgs[idx]["bbox"]
            ux0, uy0, ux1, uy1 = min(ux0, bx0), min(uy0, by0), max(ux1, bx1), max(uy1, by1)
        union_bbox = (ux0, uy0, ux1, uy1)

        # pick a representative path: prefer an exported bitmap among parts
        rep_path = None
        rep_idx = None
        for idx in cand_idxs:
            if layout_imgs[idx]["image_path"]:
                rep_path = layout_imgs[idx]["image_path"]
                rep_idx = idx
                break

        # if none has a bitmap → rasterize the union box
        if not rep_path and raster_fallback:
            rect = _safe_rect(page, union_bbox)
            if rect is not None:
                out_path = img_dir / f"{stem}_p{pno}_fig{len(figures)+1}.png"
                try:
                    _raster_crop(page, rect, raster_dpi, out_path)
                    rep_path = str(out_path)
                except Exception as e:
                    print(f"[p{pno}] raster fail (figure union): {e!r}")

        # record the figure
        parts = [layout_imgs[i] for i in cand_idxs]
        figures.append({
            "page_no": pno,
            "caption": cap["text"],
            "bbox": union_bbox,
            "image_path": rep_path,
            "parts": parts,   # keeps child block bboxes/xrefs/paths
        })

        assigned.update(cand_idxs)

    # 4) Leftovers: images without a matched caption → keep as single-image figures
    leftovers = [i for i in range(len(layout_imgs)) if i not in assigned]
    for i in leftovers:
        ib = layout_imgs[i]
        rep_path = ib["image_path"]
        if not rep_path and raster_fallback:
            rect = _safe_rect(page, ib["bbox"])
            if rect is not None:
                out_path = img_dir / f"{stem}_p{pno}_blk{i+1}_crop.png"
                try:
                    _raster_crop(page, rect, raster_dpi, out_path)
                    rep_path = str(out_path)
                except Exception as e:
                    print(f"[p{pno}] raster fail (leftover {i+1}): {e!r}")
        figures.append({
            "page_no": pno,
            "caption": None,
            "bbox": ib["bbox"],
            "image_path": rep_path,
            "parts": [ib],
        })

    stats = {
        "layout_blocks": layout_blocks,
        "xobjects_found": xobjects_found,
        "xobjects_exported": xobjects_exported,
        "figures": len(figures),
        "with_caption": sum(1 for f in figures if f["caption"]),
        "leftovers": len(leftovers),
    }
    return figures, stats













def extract_tables_for_page(
    pdf_path: str,
    pno: int,
    workdir: Path,
    stem: str,
    page_texts: List[Dict[str, Any]],
    page=None,
    *,
    flavors: Iterable[str] = ("lattice", "stream"),
    save_csv: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    tables, stats = [], {"page": pno, "tried": [], "found": 0, "saved_csv": 0,
                         "error": None, "attempts": []}

    page_h = float(page.rect.height) if page else None
    page_w = float(page.rect.width)  if page else None

    CAPTION_RE = re.compile(r'^(Table|Tab\.?)\s*\d+[\.:]?', re.I)

    # pick the FIRST "Table N." caption on this page, if any
    caption = next((t for t in page_texts if CAPTION_RE.match(t["text"])), None)

    try:
        import camelot
        # build a list of (flavor, kwargs-description, kwargs) attempts
        attempts = []

        # 1) Caption-anchored areas (above/below, two coordinate variants)
        if caption and page_h and page_w:
            for area in _areas_from_caption_bbox(caption["bbox"], page_w, page_h, pad=6):
                attempts += [
                    ("lattice", f"area={area},line_scale=40", 
                     dict(pages=str(pno), flavor="lattice", table_areas=[area],
                          process_background=True, line_scale=40)),
                    ("stream",  f"area={area},row_tol=8,col_tol=8",
                     dict(pages=str(pno), flavor="stream", table_areas=[area],
                          row_tol=8, column_tol=8, edge_tol=50)),
                ]

        # 2) Fallback: full page (slightly inset), both flavors
        inset = 6
        if page_h and page_w:
            fullA = f"{inset},{inset},{page_w-inset},{page_h-inset}"
            fullB = f"{inset},{page_h-inset},{page_w-inset},{inset}"  # flipped
            for area in (fullA, fullB):
                attempts += [
                    ("lattice", f"full={area},line_scale=40",
                     dict(pages=str(pno), flavor="lattice", table_areas=[area],
                          process_background=True, line_scale=40)),
                    ("stream",  f"full={area},row_tol=8,col_tol=8",
                     dict(pages=str(pno), flavor="stream", table_areas=[area],
                          row_tol=8, column_tol=8, edge_tol=50)),
                ]
        else:
            attempts += [
                ("lattice", "no-area,line_scale=40",
                 dict(pages=str(pno), flavor="lattice",
                      process_background=True, line_scale=40)),
                ("stream", "no-area,row/col tol",
                 dict(pages=str(pno), flavor="stream", row_tol=8, column_tol=8, edge_tol=50)),
            ]

        # Try attempts in order until we accept something
        for flavor, desc, kwargs in attempts:
            stats["tried"].append(f"{flavor}:{desc}")
            t_all = camelot.read_pdf(pdf_path, **kwargs)
            if len(t_all) == 0:
                continue

            accepted = 0
            for i, t in enumerate(t_all, start=1):
                # light semantic checks
                if t.shape[0] < 2 or t.shape[1] < 2:
                    continue
                content = "\n".join(" ".join(r) for r in t.df.values.tolist())
                if not any(ch.isdigit() for ch in content):
                    continue

                csv_path = None
                if save_csv:
                    csv_path = workdir / f"{stem}_p{pno}_table{i}_{flavor}.csv"
                    t.to_csv(str(csv_path))
                    stats["saved_csv"] += 1

                tables.append({
                    "page_no": pno, "index": i, "engine": "camelot", "flavor": flavor,
                    "nrows": t.shape[0], "ncols": t.shape[1], "csv_path": str(csv_path) if csv_path else None
                })
                accepted += 1

            if accepted:
                stats["found"] += accepted
                break  # stop after first successful attempt

    except Exception as e:
        stats["error"] = f"Camelot failed/skipped: {e!r}"

    if stats["found"] == 0 and not stats["error"]:
        stats["error"] = "no tables detected on this page"

    return tables, stats
