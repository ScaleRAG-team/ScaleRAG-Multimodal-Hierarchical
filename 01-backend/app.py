from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from typing import List
from transformers import TextIteratorStreamer
import torch, threading

# import your core pieces
from rag_core import retrieve, build_context, model, tok, SYSTEM, GEN_CFG

app = FastAPI(title="RAG API", version="0.1")

# open CORS for dev; tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.post("/api/search")
async def api_search(body: dict):
    query: str = body.get("query", "")
    k: int = int(body.get("k", 6))
    hits = retrieve(query, k=k)
    return JSONResponse({"query": query, "hits": hits})

@app.get("/api/generate")
def api_generate(query: str = Query(...), k: int = 6, max_chars: int = 1000):
    # 1) retrieve + build prompt (same format as rag_core.generate_answer)
    chunks = retrieve(query, k=k)
    context = build_context(chunks, max_chars=max_chars)
    if not context.strip():
        return StreamingResponse(iter(["data: Not found in the given context.\n\n",
                                       "data: [END]\n\n"]), media_type="text/event-stream")

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

    # 2) transformer streamer -> SSE
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True, decode_kwargs={"skip_special_tokens": True})

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=GEN_CFG["max_new_tokens"],
        do_sample=GEN_CFG["do_sample"],
        repetition_penalty=GEN_CFG["repetition_penalty"],
        no_repeat_ngram_size=GEN_CFG["no_repeat_ngram_size"],
        eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")],
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
    )

    def produce():
        # run generation in a thread so we can iterate the streamer
        t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        t.start()
        for piece in streamer:
            yield f"data: {piece}\n\n"
        yield "data: [END]\n\n"
        t.join()

    return StreamingResponse(produce(), media_type="text/event-stream")
