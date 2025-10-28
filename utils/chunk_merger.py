"""
chunk_merger.py
----------------
Merges short textual entries in RAG JSON files while keeping multimodal items
(figures, tables, equations) separate. Produces one merged file per paper.
"""

import json, glob, os, re
from pathlib import Path

def merge_short_text_chunks(input_folder="data/rag_json", output_folder="data/rag_chunks"):
    os.makedirs(output_folder, exist_ok=True)
    total_chunks = 0

    for file_path in glob.glob(f"{input_folder}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            doc_entries = json.load(f)

        merged_chunks, current_chunk = [], None

        for entry in doc_entries:
            etype = entry.get("type", "")
            text = (entry.get("content") or "").strip()

            # Merge small text-like elements (paragraphs, sections, headers)
            if etype in ["paragraph", "section", "header"]:
                is_short = len(text) < 50 or text.count(" ") < 10
                ends_with_punct = text.endswith((".", "?", "!"))

                if is_short and not ends_with_punct:
                    if current_chunk and current_chunk["type"] == "paragraph":
                        current_chunk["content"] += " " + text
                    else:
                        current_chunk = {
                            "id": entry.get("id", ""),
                            "type": "paragraph",
                            "content": text,
                            "metadata": entry.get("metadata", {}).copy(),
                        }
                    continue

                # finalize previous chunk if any
                if current_chunk:
                    current_chunk["content"] = re.sub(r"\s+", " ", current_chunk["content"]).strip()
                    merged_chunks.append(current_chunk)
                    current_chunk = None

                merged_chunks.append({
                    "id": entry.get("id", ""),
                    "type": "paragraph",
                    "content": text,
                    "metadata": entry.get("metadata", {}).copy(),
                })
            else:
                # finalize any open text chunk, then keep figure/table/equation as-is
                if current_chunk:
                    current_chunk["content"] = re.sub(r"\s+", " ", current_chunk["content"]).strip()
                    merged_chunks.append(current_chunk)
                    current_chunk = None
                merged_chunks.append(entry)

        # flush last chunk if still open
        if current_chunk:
            current_chunk["content"] = re.sub(r"\s+", " ", current_chunk["content"]).strip()
            merged_chunks.append(current_chunk)

        # write merged file per document
        output_path = Path(output_folder) / Path(file_path).name.replace(".json", ".chunks.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_chunks, f, indent=2, ensure_ascii=False)

        total_chunks += len(merged_chunks)
        print(f" +++ {Path(file_path).name}: {len(merged_chunks)} chunks saved to {output_path.name}")

    print(f"\nTotal merged chunks across all files: {total_chunks}")
    return total_chunks

