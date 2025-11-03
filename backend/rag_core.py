# rag_core.py
import torch
import pickle
import numpy as np
import faiss
import ujson as json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# -------------------- paths --------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR   = REPO_ROOT / "data" / "RAG_IS"
EMB_DIR    = DATA_DIR / "rag_embeddings"
INDEX_PATH = DATA_DIR / "rag.index"
META_PATH  = DATA_DIR / "rag.meta.pkl"




# -------------------- FAISS + metadata --------------------
index = faiss.read_index(str(INDEX_PATH))

# Load metadata rows (WITHOUT embeddings to save RAM)
meta = []
for jf in sorted(EMB_DIR.glob("*.jsonl")):
    with jf.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # drop 'embedding' if present
            rec.pop("embedding", None)
            meta.append(rec)

print(f"Loaded index with {index.ntotal} vectors; metadata rows: {len(meta)}")
if len(meta) != index.ntotal:
    print(f"[WARN] meta rows ({len(meta)}) != index.ntotal ({index.ntotal})")





# -------------------- query encoder --------------------
encoder = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

def retrieve(query: str, k: int = 10):
    q_vec = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    # safety: ensure normalized for cosine/IP
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, k)
    results = []
    for s, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        item = meta[int(i)]
        results.append({"score": float(s), **item})
    return results



# -------------------- generator --------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id  # avoid pad issues

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    low_cpu_mem_usage=True,
)

SYSTEM = (
  "You are a meticulous research assistant.\n"
  "Answer ONLY using the provided <chunk> context.\n"
  "Go through EVERY chunk.\n"
  "Step 1: Extract EVERY distinct method/model by its PROPER NAME (exact capitalization) "
  "and give a one-line description.\n"
  "Step 2: Write a concise answer that covers each named item.\n"
  "If nothing relevant: Not found in the given context.\n"
  "Paraphrase descriptions; DO NOT rename methods. Cite as [DOC:doc_id, p:page]."
)

# Keep decoding deterministic (sampling flags removed to avoid backend warnings)
GEN_CFG = dict(
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.01,
    no_repeat_ngram_size=8,
)

def build_context(chunks, max_chars: int = 1000) -> str:
    blocks = []
    for c in chunks:
        text = (c.get("text_for_embedding") or c.get("text") or "")[:max_chars]
        blocks.append(
            f"<chunk doc_id='{c.get('doc_id')}' page='{c.get('page')}' type='{c.get('type')}'>\n{text}\n</chunk>"
        )
    return "\n".join(blocks)

def generate_answer(query: str, context: str, max_new_tokens: int = GEN_CFG["max_new_tokens"]) -> str:
    if not context.strip():
        return "Not found in the given context."
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Question: {query}\n\n"
        "Use the <chunk> blocks below; paraphrase and cite as [DOC:doc_id, p:page].\n\n"
        f"{context}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=GEN_CFG["do_sample"],
        repetition_penalty=GEN_CFG["repetition_penalty"],
        no_repeat_ngram_size=GEN_CFG["no_repeat_ngram_size"],
        eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")],
        pad_token_id=tok.pad_token_id,
    )
    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids[0], skip_special_tokens=True).strip()
