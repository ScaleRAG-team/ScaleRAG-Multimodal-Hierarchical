# rag_core.py
import os
import time
import faiss
import ujson as json
import logging
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

# -------------------- logger --------------------
logger = logging.getLogger("scalerag.rag_core")

# -------------------- paths --------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "RAG_IS"
EMB_DIR = DATA_DIR / "rag_embeddings"
INDEX_PATH = DATA_DIR / "rag.index"

# -------------------- FAISS + metadata --------------------
index = faiss.read_index(str(INDEX_PATH))

meta = []
for jf in sorted(EMB_DIR.glob("*.jsonl")):
    with jf.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec.pop("embedding", None)
            meta.append(rec)

logger.info(
    "faiss_index_loaded",
    extra={"index_size": index.ntotal, "metadata_rows": len(meta)},
)

# -------------------- query encoder --------------------
encoder = SentenceTransformer(
    "Alibaba-NLP/gte-large-en-v1.5",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

def retrieve(query: str, k: int = 10):
    q_vec = encoder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, k)

    out = []
    for s, i in zip(scores[0], idxs[0]):
        if i >= 0:
            out.append({"score": float(s), **meta[int(i)]})
    return out

# -------------------- model (vLLM and AWQ) --------------------
MODEL_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

llm = LLM(
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    quantization="awq",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    trust_remote_code=True,
)

SYSTEM = (
    "You are a meticulous research assistant.\n"
    "Answer ONLY using the provided <chunk> context.\n"
    "List the relevant methods/models by their proper names.\n"
    "Write a concise answer. Cite as [DOC:doc_id, p:page]."
)

MAX_CTX_TOKENS = 1800
MAX_PROMPT_TOKENS = 4096
MAX_NEW_TOKENS = 256

def build_context(chunks):
    blocks, used = [], 0
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            continue

        block = (
            f"<chunk doc_id='{c.get('doc_id')}' page='{c.get('page')}'>\n"
            f"{text}\n</chunk>\n"
        )

        t = len(tok(block, add_special_tokens=False)["input_ids"])
        if used + t > MAX_CTX_TOKENS:
            break

        blocks.append(block)
        used += t

    return "".join(blocks)

def generate_answer(query: str, context: str) -> str:
    if not context:
        return "Not found in the given context."

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{query}\n\n{context}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > MAX_PROMPT_TOKENS:
        ids = ids[-MAX_PROMPT_TOKENS:]
        prompt = tok.decode(ids)

    params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
    )

    out = llm.generate([prompt], params)
    return out[0].outputs[0].text.strip()


# -------------------- streaming logic for vLLM --------------------
def rag_stream(query: str):
    chunks = retrieve(query, k=10)
    context = build_context(chunks)

    if not context:
        yield "Not found in the given context."
        return

    prompt = _build_prompt(query, context)

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
    )

    for output in llm.generate(
        [prompt],
        sampling_params,
        stream=True,
    ):
        
        yield output.outputs[0].text
        

# -------------------- RAG pipeline --------------------
def rag_pipeline(query: str):
    t0 = time.perf_counter()

    chunks = retrieve(query, k=10)
    context = build_context(chunks)
    answer = generate_answer(query, context)

    return {
        "answer": answer,
        "contexts": chunks,
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }
