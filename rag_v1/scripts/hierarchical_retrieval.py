import numpy as np
import faiss
# Ensure SentenceTransformer or embedding model is available:
from sentence_transformers import SentenceTransformer
# import retrieval

# Global FAISS index objects
paper_index = None    # FAISS index for paper summaries
section_index = None  # FAISS index for section summaries

def build_hierarchical_indexes(embeddings_model_name: str = None, save_dir: str = "../data2/RAG/Version_V2/indexes"):
    """
    Build FAISS indexes for paper and section summaries.
    
    This function uses the sentence embedding model (SentenceTransformer) to encode all 
    paper-level and section-level summary texts into vectors. It then constructs FAISS 
    indexes for fast similarity search. Both indexes (and accompanying ID mappings) are saved 
    to disk for reuse.
    
    Parameters:
    - embeddings_model_name: Optional name/path for the embedding model to use. If not provided, 
                              it tries to reuse the same encoder as used for the original chunks.
                              (For consistency, use the same model that produced TEXT_INDEX_PATH.)
    - save_dir: Directory to save the FAISS index files and mapping files. Defaults to the existing indexes directory.
    
    Returns: None. (The built indexes are stored in global variables `paper_index` and `section_index`, 
             and are also saved to disk.)
    
    **Note:** The summaries must be generated (via build_hierarchy) before calling this. This function 
    uses the global `paper_summaries` and `section_summaries` populated by build_hierarchy().
    """
    global paper_index, section_index, paper_summaries, section_summaries, paper_to_sections, section_to_chunks
    
    # 1. Load the sentence embedding model (SentenceTransformer) for computing embeddings.
    # If an embeddings_model_name is given, use that; otherwise use the same model as original text encoder.
    if embeddings_model_name:
        encoder = SentenceTransformer(embeddings_model_name)
    else:
        # Assume there's a retrieval utility to load the original text encoder
        try:
            import retrieval
            encoder = retrieval._load_text_encoder() 
        except Exception as e:
            raise RuntimeError("Embedding model not specified and retrieval._load_text_encoder() not available.") from e
    
    # 2. Prepare data for embedding
    paper_ids = list(paper_summaries.keys())
    paper_texts = [paper_summaries[pid] for pid in paper_ids]
    section_ids = list(section_summaries.keys())
    section_texts = [section_summaries[sid] for sid in section_ids]
    
    # 3. Compute embeddings for all summaries
    # Normalize embeddings to unit length for cosine similarity, as done in original index.
    print("[build_hierarchical_indexes] Computing embeddings for summaries...")
    paper_embeds = encoder.encode(paper_texts, normalize_embeddings=True)
    section_embeds = encoder.encode(section_texts, normalize_embeddings=True)
    paper_embeds = paper_embeds.astype('float32')
    section_embeds = section_embeds.astype('float32')
    
    # 4. Build FAISS indexes (using inner product which, with normalized vectors, is equivalent to cosine similarity).
    dim = paper_embeds.shape[1]
    paper_index = faiss.IndexFlatIP(dim)    # flat index for inner-product search
    section_index = faiss.IndexFlatIP(dim)
    paper_index.add(paper_embeds)
    section_index.add(section_embeds)
    
    # 5. Save the indexes to disk for future reuse
    paper_index_path = os.path.join(save_dir, "paper.index.faiss")
    section_index_path = os.path.join(save_dir, "section.index.faiss")
    faiss.write_index(paper_index, paper_index_path)
    faiss.write_index(section_index, section_index_path)
    # Also save the ID lists to map index results back to IDs
    id_map_path = os.path.join(save_dir, "hierarchical_id_map.pkl")
    id_data = {
        "paper_id_list": paper_ids,
        "section_id_list": section_ids,
        "paper_to_sections": paper_to_sections,
        "section_to_chunks": section_to_chunks
    }
    with open(id_map_path, "wb") as f:
        pickle.dump(id_data, f)
    
    print(f"[build_hierarchical_indexes] Built FAISS indexes: {paper_index.ntotal} papers, {section_index.ntotal} sections.")
    print(f"[build_hierarchical_indexes] Indexes saved to '{paper_index_path}' and '{section_index_path}'. ID mappings saved to '{id_map_path}'.")
