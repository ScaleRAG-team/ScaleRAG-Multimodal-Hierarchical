import os
import re
import csv
import requests


MANIFEST = "core_papers.csv"     # CSV with at least a column named: arxiv_id
OUT_DIR  = "data/pdf" # Save Directory
# ===============================

# Confrim output folder exists
os.makedirs(OUT_DIR, exist_ok=True)

# simple arXiv ID check
ARXIV_RE = re.compile(r"^\d{4}\.\d{5}(?:v\d+)?$")

def pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

def download_all_papers(manifest: str = MANIFEST, out_dir: str = OUT_DIR):
    """Download all PDFs listed in the manifest CSV."""
    os.makedirs(out_dir, exist_ok=True)

    with open(manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = (row.get("arxiv_id") or "").strip()
            title = (row.get("title") or "").strip()

            # Normalize arXiv IDs like 2309.0618 -> 2309.06180
            if re.match(r"^\d{4}\.\d{4}$", aid):
                aid = aid + "0"

            if not aid:
                print("[INFO] Skipping row with empty arxiv_id")
                continue
            if not ARXIV_RE.match(aid):
                print(f"[INFO] Skipping invalid arxiv_id: {aid}  ({title})")
                continue

            pdf_path = os.path.join(out_dir, f"{aid}.pdf")

            # check if already downloaded
            if not os.path.exists(pdf_path):
                print(f"[INFO] Downloading {aid}  ({title})")
                url = pdf_url(aid)
                try:
                    response = requests.get(url, timeout=60)
                except requests.RequestException as e:
                    print(f"[INFO] Failed to download {aid}: {e}")
                    continue

                if response.status_code == 200:
                    with open(pdf_path, "wb") as file:
                        file.write(response.content)
                    print(f"[INFO] Saved: {pdf_path}")
                else:
                    print(f"[INFO] Failed to download {aid}. Status: {response.status_code}")
            else:
                print(f"[INFO] File {pdf_path} already exists.")
