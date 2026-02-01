# app.py
import time
import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool

from rag_core import rag_pipeline. rag_stream

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
# Resource
# -------------------------
resource = Resource.create({"service.name": "scalerag-backend"})


# -------------------------
# Tracing
# -------------------------
trace.set_tracer_provider(TracerProvider(resource=resource))
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
tracer = trace.get_tracer(__name__)


# -------------------------
# Metrics
# -------------------------
metrics.set_meter_provider(MeterProvider(resource=resource,metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],))
meter = metrics.get_meter(__name__)

rag_latency_ms = meter.create_histogram(
    "rag_latency_ms",
    unit="ms",
    description="End-to-end RAG latency",
)


# -------------------------
# Logs
# -------------------------
logs.set_logger_provider(LoggerProvider(resource=resource))
logs.get_logger_provider().add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))

otel_handler = LoggingHandler(level=logging.INFO)
logging.basicConfig(level=logging.INFO, handlers=[otel_handler])
logger = logging.getLogger("scalerag")



# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="ScaleRAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Concurrency guard (important)
# -------------------------
MAX_CONCURRENT_RAG = 16  
rag_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RAG)


# -------------------------
# Health check
# -------------------------
@app.get("/healthz")
def healthz():
    logger.info("health_check")
    return PlainTextResponse("ok")


# -------------------------
# RAG generate endpoint
# -------------------------

@app.post("/api/generate")
async def api_generate(body: dict):
    
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query missing"}, status_code=400)

    with tracer.start_as_current_span("rag.request"):
        
        start = time.perf_counter()
        logger.info("rag_request", extra={"query": query})

        async with rag_semaphore:
            result = await run_in_threadpool(rag_pipeline, query)

        latency_ms = (time.perf_counter() - start) * 1000
        rag_latency_ms.record(latency_ms, {"endpoint": "generate"})

        logger.info(
            "rag_completed",
            extra={"latency_ms": latency_ms},
        )

        return JSONResponse(result)



@app.post("/api/generate_stream")
async def api_generate_stream(body: dict):
    
    query = body.get("query", "").strip()
    
    if not query:
        return JSONResponse({"error": "query missing"}, status_code=400)

    async def event_gen():
        async with rag_semaphore:
            for chunk in rag_stream(query):
                yield f"data: {chunk}\n\n"
        yield "data: [END]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
