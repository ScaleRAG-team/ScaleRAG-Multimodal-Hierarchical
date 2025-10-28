# utils/docling_converter.py
from pathlib import Path
import json
from docling.document_converter import DocumentConverter

def batch_convert_pdfs(
    in_dir: str | Path,
    out_dir: str | Path,
    verbose: bool = True
):
    """
    Convert all PDFs in a directory into Docling JSON files.

    Args:
        in_dir (str | Path): Input folder containing PDF files.
        out_dir (str | Path): Output folder for JSON files.
        verbose (bool): Print progress if True.

    Returns:
        dict: Summary containing counts of successes and failures.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    pdf_paths = sorted([p for p in in_dir.glob("*.pdf") if p.is_file()])

    if verbose:
        print(f"Found {len(pdf_paths)} PDFs in {in_dir.resolve()}")

    ok, fail = 0, 0
    for pdf_path in pdf_paths:
        try:
            res = converter.convert(str(pdf_path))
            doc = res.document
            if not doc:
                raise RuntimeError("Docling returned no document")

            doc_dict = doc.export_to_dict()
            out_path = out_dir / f"{pdf_path.stem}.docling.json"

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)

            ok += 1
            if verbose:
                print(f" +++ Saved: {out_path.name}")
        except Exception as e:
            fail += 1
            if verbose:
                print(f" --- Failed on {pdf_path.name}: {e}")

    summary = {"success": ok, "failed": fail}
    if verbose:
        print(f"\nDone. Success: {ok}, Failed: {fail}")
        print(f"JSONs saved in: {out_dir.resolve()}")
    return summary
