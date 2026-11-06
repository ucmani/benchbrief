import logging
import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from observability.logging_config import configure_logging, new_request_id
from observability.metrics import REQUEST_LATENCY, REQUEST_COUNT

from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool

from generation.pipeline import run_pipeline
from generation.chat import answer_question
from utils.llm import make_llm_chain, describe_chain

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="BenchBrief")


def llm_specs() -> str:
    return os.environ.get("GEN_LLM", "gemini,openrouter")


@lru_cache(maxsize=1)
def get_llm():
    """One shared LLM function, built on first use. GEN_LLM picks the providers,
    e.g. "gemini,openrouter=<model id>" (first = primary, rest = fallbacks)."""
    logger.info(f"LLM chain: {describe_chain(llm_specs())}")
    return make_llm_chain(llm_specs())


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = new_request_id()
    logger.info(f"{request.method} {request.url.path} started")
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} completed in {duration:.3f}s, status={response.status_code}")
    response.headers["X-Request-Id"] = request_id
    return response

class CaseRequest(BaseModel):
    current_case_text: str
    top_k: int = 5
    judge: str | None = None

class CaseResponse(BaseModel):
    case_brief: str
    recommendation: str
    precedent_chunks: list[dict]

class ChatMessage(BaseModel):
    role: str          # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]   # whole conversation, last one is the new question
    top_k: int = 8

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.post("/analyze", response_model = CaseResponse)
async def analyze_case(request: CaseRequest) -> CaseResponse:
    result = await run_in_threadpool(
        run_pipeline,
        current_case_text=request.current_case_text,
        llm_call_fn=get_llm(),
        top_k=request.top_k,
        judge=request.judge,
    )
    return CaseResponse(**result)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="last message must be from the user")
    result = await run_in_threadpool(
        answer_question,
        messages=[m.model_dump() for m in request.messages],
        llm_call_fn=get_llm(),
        top_k=request.top_k,
    )
    return ChatResponse(**result)

@app.get("/info")
async def info() -> dict:
    return {"generation": describe_chain(llm_specs()),   # first = primary, rest = fallbacks
            "rerank": os.environ.get("USE_RERANK", "1") == "1"}

@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)