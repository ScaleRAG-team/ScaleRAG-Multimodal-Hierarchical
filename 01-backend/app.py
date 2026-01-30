from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from typing import List
from transformers import TextIteratorStreamer
import torch, threading

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

resource = Resource.create({"service.name": "scalerag-backend"})

# Tracing
trace.set_tracer_provider(TracerProvider(resource=resource))
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)
tracer = trace.get_tracer(__name__)

# Metrics
metrics.set_meter_provider(
    MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(OTLPMetricExporter())
        ],
    )
)
meter = metrics.get_meter(__name__)

# Metrics we care about (for now)
rag_latency_ms = meter.create_histogram(
    "rag_latency_ms",
    unit="ms",
    description="End-to-end RAG latency"
)


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
    with tracer.start_as_current_span("rag.search"):
        start = time.time()

        query: str = body.get("query", "")
        k: int = int(body.get("k", 6))
        hits = retrieve(query, k=k)

        rag_latency_ms.record(
            (time.time() - start) * 1000,
            {"endpoint": "search"}
        )

        return JSONResponse({"query": query, "hits": hits})



@app.get("/api/generate")
def api_generate(query: str = Query(...), k: int = 6, max_chars: int = 1000):
    with tracer.start_as_current_span("rag.generate"):
        start = time.time()

        # 1) retrieve + build prompt
        chunks = retrieve(query, k=k)
        context = build_context(chunks, max_chars=max_chars)

        if not context.strip():
            rag_latency_ms.record(
                (time.time() - start) * 1000,
                {"endpoint": "generate"}
            )
            return StreamingResponse(
                iter([
                    "data: Not found in the given context.\n\n",
                    "data: [END]\n\n"
                ]),
                media_type="text/event-stream"
            )

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

        streamer = TextIteratorStreamer(
            tok,
            skip_special_tokens=True,
            decode_kwargs={"skip_special_tokens": True}
        )

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
            t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            t.start()
            for piece in streamer:
                yield f"data: {piece}\n\n"
            yield "data: [END]\n\n"
            t.join()

            rag_latency_ms.record(
                (time.time() - start) * 1000,
                {"endpoint": "generate"}
            )

        return StreamingResponse(produce(), media_type="text/event-stream")
