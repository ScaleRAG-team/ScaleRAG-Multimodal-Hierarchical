from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from transformers import TextIteratorStreamer
import torch, threading, time, logging

# -------------------------
# OpenTelemetry imports
# -------------------------
from opentelemetry import trace, metrics, logs
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter


# -------------------------
# Resource (shared)
# -------------------------
resource = Resource.create({
    "service.name": "scalerag-backend"
})


# -------------------------
# Tracing
# -------------------------
trace.set_tracer_provider(TracerProvider(resource=resource))
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)
tracer = trace.get_tracer(__name__)


# -------------------------
# Metrics
# -------------------------
metrics.set_meter_provider(
    MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(OTLPMetricExporter())
        ],
    )
)
meter = metrics.get_meter(__name__)

rag_latency_ms = meter.create_histogram(
    "rag_latency_ms",
    unit="ms",
    description="End-to-end RAG latency"
)


# -------------------------
# Logs (Loki via OTel)
# -------------------------
logs.set_logger_provider(LoggerProvider(resource=resource))
logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)

otel_handler = LoggingHandler(level=logging.INFO)
logging.basicConfig(level=logging.INFO, handlers=[otel_handler])
logger = logging.getLogger("scalerag")


# -------------------------
# Import core pieces
# -------------------------
from rag_core import retrieve, build_context, model, tok, SYSTEM, GEN_CFG


# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="RAG API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    logger.info("health_check")
    return PlainTextResponse("ok")


@app.post("/api/search")
async def api_search(body: dict):
    with tracer.start_as_current_span("rag.search"):
        start = time.time()

        query: str = body.get("query", "")
        k: int = int(body.get("k", 6))

        logger.info(
            "search_request",
            extra={"query": query, "k": k}
        )

        hits = retrieve(query, k=k)

        rag_latency_ms.record(
            (time.time() - start) * 1000,
            {"endpoint": "search"}
        )

        logger.info(
            "search_completed",
            extra={"num_hits": len(hits)}
        )

        return JSONResponse({"query": query, "hits": hits})


@app.get("/api/generate")
def api_generate(query: str = Query(...), k: int = 6, max_chars: int = 1000):
    with tracer.start_as_current_span("rag.generate"):
        start = time.time()

        logger.info(
            "generate_request",
            extra={"query": query, "k": k}
        )

        chunks = retrieve(query, k=k)
        context = build_context(chunks, max_chars=max_chars)

        if not context.strip():
            logger.warning(
                "empty_context",
                extra={"query": query}
            )

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

            logger.info(
                "generation_completed",
                extra={"latency_ms": (time.time() - start) * 1000}
            )

        return StreamingResponse(produce(), media_type="text/event-stream")
