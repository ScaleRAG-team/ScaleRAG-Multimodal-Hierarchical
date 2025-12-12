import os
import json
import pickle
import faiss

# Configure paths
INDEX_DIR = "../data2/RAG/Version_V2/indexes"
EMBEDDINGS_PATH = "../data2/RAG/Version_V2/embeddings/all_papers.embeddings.pkl"
TEXT_INDEX_PATH = os.path.join(INDEX_DIR, "text.index.faiss")
IMAGE_INDEX_PATH = os.path.join(INDEX_DIR, "image.index.faiss")

# Global variables to hold loaded data structures
text_index = None
image_index = None
id_to_text = {}    # Mapping from text chunk ID -> text content
id_to_meta = {}    # Mapping from chunk ID -> metadata (section, paper, etc.)
id_to_image = {}   # Mapping from image ID -> image file path (for figure images)

def load_corpus(rag_chunks_dir: str):
    """
    Load RAG chunk JSON files from the specified directory.
    Populates id_to_text, id_to_meta, and id_to_image dictionaries.
    """
    global id_to_text, id_to_meta, id_to_image
    id_to_text.clear()
    id_to_meta.clear()
    id_to_image.clear()
    files = [f for f in os.listdir(rag_chunks_dir) if f.endswith(".json")]
    for fname in files:
        fpath = os.path.join(rag_chunks_dir, fname)
        with open(fpath, 'r') as f:
            chunks = json.load(f)
            for chunk in chunks:
                cid = chunk["id"]
                ctype = chunk["type"]
                if ctype == "paragraph" or ctype == "figure":
                    # For text chunks (paragraphs or figure captions)
                    id_to_text[cid] = chunk["content"]
                    id_to_meta[cid] = chunk.get("metadata", {})
                    # If figure, also record image path if present
                    if ctype == "figure":
                        img_path = chunk["metadata"].get("image_path")
                        if img_path:
                            id_to_image[cid] = img_path
                else:
                    # Handle other types if any (e.g., tables or equations)
                    id_to_text[cid] = chunk["content"]
                    id_to_meta[cid] = chunk.get("metadata", {})
    print(f"[load_corpus] Loaded {len(id_to_text)} text/figure chunks from {len(files)} files.")

def load_faiss_indexes():
    """Load Faiss indexes for text and image embeddings from disk."""
    global text_index, image_index
    # Load text index
    if os.path.exists(TEXT_INDEX_PATH):
        text_index = faiss.read_index(TEXT_INDEX_PATH)
        print("[load_faiss_indexes] Text index loaded.")
    else:
        raise FileNotFoundError(f"Text index not found at {TEXT_INDEX_PATH}")
    # Load image index
    if os.path.exists(IMAGE_INDEX_PATH):
        image_index = faiss.read_index(IMAGE_INDEX_PATH)
        print("[load_faiss_indexes] Image index loaded.")
    else:
        raise FileNotFoundError(f"Image index not found at {IMAGE_INDEX_PATH}")

def load_embeddings():
    """
    Load pre-computed embeddings (if needed for mapping or building hierarchical indexes).
    Returns a dict with keys like 'text_embeds', 'text_ids', 'image_embeds', 'image_ids'.
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}")
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    print("[load_embeddings] Loaded embeddings from pickle.")
    return embeddings

def get_text_content(chunk_id: str) -> str:
    """Retrieve the text content for a given chunk ID."""
    return id_to_text.get(chunk_id, "")

def get_metadata(chunk_id: str) -> dict:
    """Retrieve metadata (e.g., section, page, etc.) for a given chunk ID."""
    return id_to_meta.get(chunk_id, {})

def get_image_path(image_id: str) -> str:
    """Retrieve the file path for a figure given its chunk ID."""
    return id_to_image.get(image_id, "")
