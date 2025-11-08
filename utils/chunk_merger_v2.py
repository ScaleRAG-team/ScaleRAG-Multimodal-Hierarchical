"""
chunk_merger_v2.py
------------------
Second iteration of the chunk merger.

- Two-pass merge (merge then cleanup)
- Drop/ignore author + affiliation lines on the first page (common in arXiv PDFs)
- Keep abstract/introduction
- Mark code-looking chunks (e.g. "R> ...") as non-rankable
- Write to data/rag_chunks_v2
"""

import json, glob, os, re
from pathlib import Path
from typing import List, Dict, Any

TEXT_TYPES = {"paragraph", "section", "header", "text"}
MAX_MERGED_TOKENS = 220
INPUT_FOLDER = "data/rag_json"
OUTPUT_FOLDER = "data/rag_chunks_v2"

# words that often appear in affiliation lines
AFFIL_WORDS = {
    "university", "institute", "laboratory", "lab", "department",
    "apple", "google", "meta", "openai", "microsoft", "amazon",
    "school", "college", "research", "ai", "†", "‡"
}

JUNK_PATTERNS = [
    r"equal contribution",
    r"correspondence to",
    r"@.+\..+",
    r"^\d{4}\s*$",
]

NAME_RE = re.compile(r"^[A-Z][a-z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-z]+)+$")


def is_probably_name_line(text: str) -> bool:
    return bool(NAME_RE.match(text.strip()))


def looks_like_author_affiliation(text: str, metadata: Dict[str, Any]) -> bool:
    """
    Heuristic for page-1 author blocks like:
    'Keivan Alizadeh, Iman Mirzadeh *, ... Apple †'
    We'll only apply this on page 1.
    """
    page = metadata.get("page")
    if page not in (0, 1):  # some pdf extractors start at 0, others at 1
        return False

    t = text.strip()
    # lots of commas and no verb → likely authors
    comma_count = t.count(",")
    if comma_count >= 3:
        lower = t.lower()
        if any(w in lower for w in AFFIL_WORDS):
            return True
        # many capitalized name-like tokens
        if is_probably_name_line(t.replace(",", " ")):
            return True
    return False


def is_code_like(text: str) -> bool:
    # detect common code/prompt-looking lines and mark them as non-rankable
    stripped = text.lstrip()
    if stripped.startswith("R> ") or stripped.startswith(">>> "):
        return True
    # looks like a shell / code prompt
    if re.match(r"^[a-zA-Z_]+\s*<-", stripped):
        return True
    return False


def is_junk_text(text: str, metadata: Dict[str, Any]) -> bool:
    t = text.strip()
    if not t:
        return True
    tl = t.lower()

    # page-1 author/affil killer
    if looks_like_author_affiliation(t, metadata):
        return True

    for pat in JUNK_PATTERNS:
        if re.search(pat, tl):
            return True
    if is_probably_name_line(t):
        return True
    if tl in {"with", "defining", "and", "or"}:
        return True
    return False


def count_tokens_like(text: str) -> int:
    return len(text.split())


def same_location(a_meta: Dict[str, Any], b_meta: Dict[str, Any]) -> bool:
    if not a_meta and not b_meta:
        return True
    if a_meta.get("page") is not None and b_meta.get("page") is not None:
        if a_meta["page"] != b_meta["page"]:
            return False
    if a_meta.get("section") and b_meta.get("section"):
        if a_meta["section"] != b_meta["section"]:
            return False
    return True


def looks_like_garbled_equation(text: str) -> bool:
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) > 0.4


def flush_current(buffer: List[Dict[str, Any]], out: List[Dict[str, Any]]):
    if not buffer:
        return
    contents = [b["content"] for b in buffer if b["content"].strip()]
    merged_text = re.sub(r"\s+", " ", " ".join(contents)).strip()
    if not merged_text:
        buffer.clear()
        return
    base = buffer[0]
    chunk = {
        "id": base.get("id", ""),
        "type": "paragraph",
        "content": merged_text,
        "metadata": base.get("metadata", {}).copy(),
        "source_ids": [b.get("id", "") for b in buffer],
    }
    # if any piece was code-like, we can mark whole thing as non-rankable
    if any(b.get("_code_like") for b in buffer):
        chunk["rankable"] = False
    out.append(chunk)
    buffer.clear()


def pass1_merge(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    for entry in entries:
        etype = entry.get("type", "").lower()
        text = (entry.get("content") or "").strip()
        metadata = entry.get("metadata", {}) or {}

        if etype in TEXT_TYPES:
            if is_junk_text(text, metadata):
                flush_current(current, merged)
                continue

            is_code = is_code_like(text)

            tokens = count_tokens_like(text)
            if not current:
                current.append(
                    {
                        "id": entry.get("id", ""),
                        "content": text,
                        "metadata": metadata,
                        "_code_like": is_code,
                    }
                )
            else:
                buf_tokens = count_tokens_like(" ".join(b["content"] for b in current))
                if same_location(current[0]["metadata"], metadata) and (
                    buf_tokens + tokens <= MAX_MERGED_TOKENS
                ):
                    current.append(
                        {
                            "id": entry.get("id", ""),
                            "content": text,
                            "metadata": metadata,
                            "_code_like": is_code,
                        }
                    )
                else:
                    flush_current(current, merged)
                    current.append(
                        {
                            "id": entry.get("id", ""),
                            "content": text,
                            "metadata": metadata,
                            "_code_like": is_code,
                        }
                    )

        elif etype == "equation":
            flush_current(current, merged)

            if merged and not looks_like_garbled_equation(text):
                last = merged[-1]
                if last.get("type") == "paragraph" and same_location(
                    last.get("metadata", {}), metadata
                ):
                    last["content"] += f"\nEquation: {text}"
                    src = last.get("source_ids", [])
                    src.append(entry.get("id", ""))
                    last["source_ids"] = src
                    continue

            eq_entry = dict(entry)
            if looks_like_garbled_equation(text):
                eq_entry["rankable"] = False
            merged.append(eq_entry)

        else:
            flush_current(current, merged)
            merged.append(entry)

    flush_current(current, merged)
    return merged


def pass2_cleanup(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for ch in chunks:
        if (
            cleaned
            and ch.get("type") == "paragraph"
            and len(ch.get("content", "")) < 80
            and count_tokens_like(ch.get("content", "")) < 12
        ):
            prev = cleaned[-1]
            if prev.get("type") == "paragraph" and same_location(
                prev.get("metadata", {}), ch.get("metadata", {})
            ):
                prev["content"] = (prev["content"] + " " + ch["content"]).strip()
                src = prev.get("source_ids", [])
                src.extend(ch.get("source_ids", [ch.get("id", "")]))
                prev["source_ids"] = src
                # if the short one was non-rankable, propagate
                if ch.get("rankable") is False:
                    prev["rankable"] = False
                continue
        cleaned.append(ch)
    return cleaned


def merge_short_text_chunks_v2(
    input_folder: str = INPUT_FOLDER, output_folder: str = OUTPUT_FOLDER
) -> int:
    os.makedirs(output_folder, exist_ok=True)
    total = 0
    for file_path in glob.glob(f"{input_folder}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        p1 = pass1_merge(entries)
        final_chunks = pass2_cleanup(p1)

        out_name = Path(file_path).name.replace(".json", ".chunks.json")
        out_path = Path(output_folder) / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, indent=2, ensure_ascii=False)

        print(f"[v2] {Path(file_path).name}: {len(final_chunks)} chunks -> {out_path.name}")
        total += len(final_chunks)

    print(f"\n[v2] Total merged chunks across all files: {total}")
    return total


if __name__ == "__main__":
    merge_short_text_chunks_v2()
