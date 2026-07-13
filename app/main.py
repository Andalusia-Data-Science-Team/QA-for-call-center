import time
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.models.input import CallTranscript, BatchCallTranscripts
from app.models.output import QAAnalysisResult, BatchQAAnalysisResult
from app.agent import QAAgent
from app.services.llm_client import LLMClient
from app.config import settings

TEMPLATES_DIR = Path(__file__).parent / "templates"
CM_DIR        = Path(__file__).parent / "CM"
SQL_DIR       = Path(__file__).parent / "SQL"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Call QA Analysis API [LangGraph] (provider=%s)", settings.llm_provider)
    yield
    logger.info("Shutting down Call QA Analysis API")


app = FastAPI(
    title="Call QA Analysis API",
    description="AI-powered quality analysis for clinical call center transcripts.",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

llm_client = LLMClient(provider=settings.llm_provider, model=settings.llm_model)
analyzer   = QAAgent(llm_client=llm_client)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response


# ── Static pages ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "provider": settings.llm_provider, "model": settings.llm_model}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="qa-dashboard.html")


@app.get("/qa-dashboard.html", response_class=HTMLResponse)
async def qa_dashboard_page(request: Request):
    return templates.TemplateResponse(request=request, name="qa-dashboard.html")


@app.get("/agents-dashboard.html", response_class=HTMLResponse)
async def agents_dashboard_page(request: Request):
    return templates.TemplateResponse(request=request, name="agents-dashboard.html")


# ── Agent email list ──────────────────────────────────────────────────────────

@app.get("/agents/emails")
async def list_agent_emails(
    json_file: str = Query(default="passcode.json"),
    db_key:    str = Query(default="CM"),
):
    import sys
    sys.path.insert(0, str(CM_DIR))
    from CM_DB_Handler import CMDatabaseHandler

    # Resolve json_file relative to CM_DIR if not absolute
    json_path = Path(json_file)
    if not json_path.is_absolute():
        json_path = CM_DIR / json_file
    if not json_path.exists():
        raise HTTPException(status_code=500, detail=f"Credentials file not found: {json_path}")

    handler = CMDatabaseHandler(json_file=str(json_path), db_key=db_key)
    if not handler.connect():
        raise HTTPException(status_code=503, detail="Cannot connect to CM database.")

    sql_file = str(SQL_DIR / "CM_users.sql")
    try:
        df = handler.execute_query_from_file(sql_file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
    finally:
        handler.close()

    emails = (
        df["EmailAddress"]
        .dropna()
        .unique()
        .tolist()
    )
    emails.sort()
    return {"emails": emails}


# ── Retrieve + Analyze pipeline ───────────────────────────────────────────────

@app.post("/retrieve-and-analyze", response_model=BatchQAAnalysisResult)
async def retrieve_and_analyze(
    agent_email: str = Query(..., description="Agent email address"),
    filter_date: str = Query(..., description="Date in YYYY-MM-DD format"),
    json_file:   str = Query(default="passcode.json"),
    db_key:      str = Query(default="CM"),
):
    import asyncio, sys
    sys.path.insert(0, str(CM_DIR))
    from CM_DB_Handler import CMDatabaseHandler
    from Chat_Retriever import retrieve_chats_by_agent_email, df_to_batch

    # Resolve json_file relative to CM_DIR if not absolute
    json_path = Path(json_file)
    if not json_path.is_absolute():
        json_path = CM_DIR / json_file
    if not json_path.exists():
        raise HTTPException(status_code=500, detail=f"Credentials file not found: {json_path}")

    logger.info("retrieve-and-analyze | agent=%s date=%s", agent_email, filter_date)

    # ── Step 1: fetch rows ────────────────────────────────────────────────────
    handler = CMDatabaseHandler(json_file=str(json_path), db_key=db_key)
    try:
        df = retrieve_chats_by_agent_email(
            agent_email=agent_email,
            filter_date=filter_date,
            db_handler=handler,
            sql_file=str(SQL_DIR / "CM_Chat_Search.sql"),
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("retrieve-and-analyze: DB error")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No chats found for {agent_email} on {filter_date}.",
        )

    logger.info(
        "retrieve-and-analyze | %d rows / %d conversations retrieved",
        len(df), df["UniqueId"].nunique(),
    )

    # ── Step 2: convert to BatchCallTranscripts ───────────────────────────────
    try:
        batch = df_to_batch(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Transcript conversion failed: {exc}")

    logger.info("retrieve-and-analyze | %d calls passed validation", len(batch.calls))

    # ── Step 3: analyze each call ─────────────────────────────────────────────
    async def safe_analyze(call: CallTranscript):
        try:
            return await analyzer.analyze(call)
        except Exception as exc:
            logger.error("analyze failed | call_id=%s | %s", call.call_id, exc)
            return QAAnalysisResult.error_result(call.call_id, str(exc))

    results = await asyncio.gather(*[safe_analyze(c) for c in batch.calls])

    summary = {
        "total":        len(results),
        "pass":         sum(1 for r in results if r.overall_assessment == "pass"),
        "needs_review": sum(1 for r in results if r.overall_assessment == "needs_review"),
        "escalate":     sum(1 for r in results if r.overall_assessment == "escalate"),
        "errors":       sum(1 for r in results if r.overall_assessment == "error"),
    }
    logger.info("retrieve-and-analyze done | summary=%s", summary)
    return BatchQAAnalysisResult(results=list(results), summary=summary)


# ── Existing single / batch endpoints (unchanged) ─────────────────────────────

@app.post("/upload-analyze", response_model=QAAnalysisResult)
async def upload_analyze(file: UploadFile = File(...)):
    try:
        raw  = await file.read()
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}")

    try:
        payload = CallTranscript(**data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"JSON does not match CallTranscript schema: {exc}")

    logger.info("upload-analyze | call_id=%s agent=%s dept=%s", payload.call_id, payload.agent_name, payload.department)
    try:
        result = await analyzer.analyze(payload)
    except Exception as exc:
        logger.exception("upload-analyze failed | call_id=%s", payload.call_id)
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}")

    logger.info("upload-analyze done | call_id=%s assessment=%s", payload.call_id, result.overall_assessment)
    return result


@app.post("/analyze-call", response_model=QAAnalysisResult)
async def analyze_call(payload: CallTranscript) -> QAAnalysisResult:
    logger.info("analyze-call | call_id=%s agent=%s dept=%s duration=%ss",
                payload.call_id, payload.agent_name, payload.department, payload.call_duration_seconds)
    try:
        result = await analyzer.analyze(payload)
    except Exception as exc:
        logger.exception("analyze-call failed | call_id=%s", payload.call_id)
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}")

    logger.info("analyze-call done | call_id=%s assessment=%s", payload.call_id, result.overall_assessment)
    return result


@app.post("/batch-analyze", response_model=BatchQAAnalysisResult)
async def batch_analyze(payload: BatchCallTranscripts) -> BatchQAAnalysisResult:
    import asyncio
    logger.info("batch-analyze | count=%d", len(payload.calls))

    async def safe_analyze(call: CallTranscript):
        try:
            return await analyzer.analyze(call)
        except Exception as exc:
            logger.error("batch item failed | call_id=%s | %s", call.call_id, exc)
            return QAAnalysisResult.error_result(call.call_id, str(exc))

    results = await asyncio.gather(*[safe_analyze(c) for c in payload.calls])
    summary = {
        "total":        len(results),
        "pass":         sum(1 for r in results if r.overall_assessment == "pass"),
        "needs_review": sum(1 for r in results if r.overall_assessment == "needs_review"),
        "escalate":     sum(1 for r in results if r.overall_assessment == "escalate"),
        "errors":       sum(1 for r in results if r.overall_assessment == "error"),
    }
    logger.info("batch-analyze done | summary=%s", summary)
    return BatchQAAnalysisResult(results=list(results), summary=summary)
