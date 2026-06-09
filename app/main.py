import time
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.models.input import CallTranscript, BatchCallTranscripts
from app.models.output import QAAnalysisResult, BatchQAAnalysisResult
from app.agent import QAAgent
from app.services.llm_client import LLMClient
from app.config import settings

TEMPLATES_DIR = Path(__file__).parent / "templates"

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Call QA Analysis API [LangGraph] (provider=%s)", settings.llm_provider)
    yield
    logger.info("Shutting down Call QA Analysis API")


# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Call QA Analysis API",
    description="AI-powered quality analysis for clinical call center transcripts.",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Build a shared LangGraph-backed QA agent (compiles graph once at startup)
llm_client = LLMClient(provider=settings.llm_provider, model=settings.llm_model)
analyzer = QAAgent(llm_client=llm_client)


# Middleware: request-level timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response

############################################################################
# Endpoints 
@app.get("/health")
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "provider": settings.llm_provider, "model": settings.llm_model}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the QA dashboard UI."""
    return templates.TemplateResponse(request=request, name="qa-dashboard.html")


@app.post("/upload-analyze", response_model=QAAnalysisResult)
async def upload_analyze(file: UploadFile = File(..., description="JSON file containing a CallTranscript")) -> QAAnalysisResult:
    """
    Accept a JSON file upload containing a CallTranscript, analyse it, and
    return the structured QA report.  Designed to be called from the browser
    dashboard.
    """
    if file.content_type not in ("application/json", "text/plain", "application/octet-stream"):
        # Be lenient — browsers may send different content-types
        pass

    try:
        raw = await file.read()
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}") from exc

    try:
        payload = CallTranscript(**data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"JSON does not match CallTranscript schema: {exc}") from exc

    logger.info(
        "upload-analyze | call_id=%s agent=%s dept=%s",
        payload.call_id,
        payload.agent_name,
        payload.department,
    )
    try:
        result = await analyzer.analyze(payload)
    except Exception as exc:
        logger.exception("upload-analyze failed | call_id=%s", payload.call_id)
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}") from exc

    logger.info(
        "upload-analyze done | call_id=%s assessment=%s escalate=%s",
        payload.call_id,
        result.overall_assessment,
        result.escalation_required,
    )
    return result


@app.post("/analyze-call", response_model=QAAnalysisResult)
async def analyze_call(payload: CallTranscript) -> QAAnalysisResult:
    """
    Analyze a single call transcript and return a structured QA report.

    - Detects HIPAA concerns, misinformation, rudeness, protocol violations,
      and positive interactions.
    - Returns agent performance scores plus escalation guidance.
    """
    logger.info(
        "analyze-call | call_id=%s agent=%s dept=%s duration=%ss",
        payload.call_id,
        payload.agent_name,
        payload.department,
        payload.call_duration_seconds,
    )
    try:
        result = await analyzer.analyze(payload)
    except Exception as exc:
        logger.exception("analyze-call failed | call_id=%s", payload.call_id)
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}") from exc

    logger.info(
        "analyze-call done | call_id=%s assessment=%s escalate=%s",
        payload.call_id,
        result.overall_assessment,
        result.escalation_required,
    )
    return result


@app.post("/batch-analyze", response_model=BatchQAAnalysisResult)
async def batch_analyze(payload: BatchCallTranscripts) -> BatchQAAnalysisResult:
    """
    Analyze a batch of call transcripts concurrently.

    Results are returned in the same order as the input list.
    Each item is independently analyzed; one failure does not block others.
    """
    import asyncio

    logger.info("batch-analyze | count=%d", len(payload.calls))

    async def safe_analyze(call: CallTranscript):
        try:
            return await analyzer.analyze(call)
        except Exception as exc:
            logger.error("batch item failed | call_id=%s | %s", call.call_id, exc)
            # Return a minimal error result rather than killing the whole batch
            return QAAnalysisResult.error_result(call.call_id, str(exc))

    results = await asyncio.gather(*[safe_analyze(c) for c in payload.calls])

    summary = {
        "total": len(results),
        "pass": sum(1 for r in results if r.overall_assessment == "pass"),
        "needs_review": sum(1 for r in results if r.overall_assessment == "needs_review"),
        "escalate": sum(1 for r in results if r.overall_assessment == "escalate"),
        "errors": sum(1 for r in results if r.overall_assessment == "error"),
    }
    logger.info("batch-analyze done | summary=%s", summary)
    return BatchQAAnalysisResult(results=list(results), summary=summary)
