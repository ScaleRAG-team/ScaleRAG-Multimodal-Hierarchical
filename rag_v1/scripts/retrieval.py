import torch
from transformers import CLIPModel, CLIPTokenizer
from sentence_transformers import SentenceTransformer, util as st_util

# Initialize encoders
_text_encoder_model = None
_image_encoder_model = None
_clip_tokenizer = None

# Load models
def _load_text_encoder():
    global _text_encoder_model
    if _text_encoder_model is None:
        _text_encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _text_encoder_model

def _load_image_encoder():
    global _image_encoder_model, _clip_tokenizer
    if _image_encoder_model is None:
        _image_encoder_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return _image_encoder_model, _clip_tokenizer

def query_text(query: str, k: int = 3, restrict_ids: list = None):
    """
    Search the text FAISS index for the query and return top-k relevant text chunks.
    If restrict_ids is provided, limit consideration to those chunk IDs (e.g., within a specific paper).
    """
    assert rag_loader.text_index is not None, "Text index is not loaded."
    model = _load_text_encoder()
    # Encode query to vector
    query_vec = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    query_vec = query_vec.cpu().numpy().astype('float32')
    # If restricting to subset, perform brute-force search on that subset
    if restrict_ids:
        # Get embeddings for restrict_ids from loaded embeddings or by encoding text
        subset_vectors = []
        subset_ids = []
        for cid in restrict_ids:
            content = rag_loader.get_text_content(cid)
            if content:
                subset_ids.append(cid)
                vec = model.encode(content, convert_to_tensor=True, normalize_embeddings=True)
                subset_vectors.append(vec.cpu().numpy().astype('float32'))
        if not subset_vectors:
            return []
        subset_matrix = np.vstack(subset_vectors)
        # Compute cosine similarity manually and get top-k
        sim = np.dot(subset_matrix, query_vec.T).squeeze()  # since vectors are L2-normalized
        top_idx = sim.argsort()[-k:][::-1]
        results = []
        for idx in top_idx:
            cid = subset_ids[idx]
            results.append({
                "id": cid,
                "score": float(sim[idx]),
                "content": rag_loader.get_text_content(cid),
                "metadata": rag_loader.get_metadata(cid)
            })
        return results
    else:
        # Perform FAISS search on full index
        D, I = rag_loader.text_index.search(query_vec.reshape(1, -1), k)
        # D: distances, I: indices
        results = []
        for score, idx in zip(D[0], I[0]):
            # FAISS may return -1 for empty results
            if idx == -1:
                continue
            if not hasattr(rag_loader, "text_id_list"):
                embeds = rag_loader.load_embeddings()
                rag_loader.text_id_list = embeds.get("text_ids", [])
            cid = rag_loader.text_id_list[idx]
            results.append({
                "id": cid,
                "score": float(score),
                "content": rag_loader.get_text_content(cid),
                "metadata": rag_loader.get_metadata(cid)
            })
        return results

def query_images(query: str, k: int = 2, restrict_paper_id: str = None):
    """
    Search the image FAISS index for relevant images (figures) given a text query.
    Uses CLIP text encoder for the query. Optionally restrict to a specific paper's images.
    """
    assert rag_loader.image_index is not None, "Image index is not loaded."
    model, tokenizer = _load_image_encoder()
    # Encode query text to CLIP text embedding
    inputs = tokenizer(query, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2)  # normalize
    text_vec = text_features.detach().cpu().numpy().astype('float32')
    # If restricting to images of one paper, filter those first
    if restrict_paper_id:
        # Gather all image IDs for that paper
        image_ids = [cid for cid in rag_loader.id_to_image.keys() if cid.startswith(restrict_paper_id)]
        # If none, return empty
        if not image_ids:
            return []
        # Brute-force search on those images: get their embeddings
        embeds = rag_loader.load_embeddings()
        all_image_ids = embeds.get("image_ids", [])
        all_image_embeds = embeds.get("image_embeds", [])
        # Filter to the ones matching restrict_paper_id
        subset_embeds = []
        subset_ids = []
        for cid, emb in zip(all_image_ids, all_image_embeds):
            if cid in image_ids:
                subset_ids.append(cid)
                subset_embeds.append(emb)
        if not subset_ids:
            return []
        subset_matrix = np.vstack(subset_embeds)
        # compute similarity (dot product since normalized)
        sim = np.dot(subset_matrix, text_vec.T).squeeze()
        top_idx = sim.argsort()[-k:][::-1]
        results = []
        for idx in top_idx:
            cid = subset_ids[idx]
            results.append({
                "id": cid,
                "score": float(sim[idx]),
                "image_path": rag_loader.get_image_path(cid),
                "caption": rag_loader.get_text_content(cid),
                "metadata": rag_loader.get_metadata(cid)
            })
        return results
    else:
        # Global image index search
        D, I = rag_loader.image_index.search(text_vec.reshape(1, -1), k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            # Map index to image ID
            if not hasattr(rag_loader, "image_id_list"):
                embeds = rag_loader.load_embeddings()
                rag_loader.image_id_list = embeds.get("image_ids", [])
            image_id = rag_loader.image_id_list[idx]
            results.append({
                "id": image_id,
                "score": float(score),
                "image_path": rag_loader.get_image_path(image_id),
                "caption": rag_loader.get_text_content(image_id),  # figure caption text
                "metadata": rag_loader.get_metadata(image_id)
            })
        return results
