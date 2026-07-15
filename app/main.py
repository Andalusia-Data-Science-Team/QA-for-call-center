import json                          # ← was missing; needed by upload_analyze
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query, Form, status
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
# StaticFiles removed — it was imported but never used
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

USER_STORE = {
    "leader": {"password": "leaderpass", "role": "team_leader"},
    "admin":  {"password": "adminpass",  "role": "qa_admin"},
}

llm_client = LLMClient(provider=settings.llm_provider, model=settings.llm_model)
analyzer   = QAAgent(llm_client=llm_client)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "provider": settings.llm_provider, "model": settings.llm_model}


# ── Auth pages ────────────────────────────────────────────────────────────────
# NOTE: all TemplateResponse calls now use the Starlette 1.x signature:
#   templates.TemplateResponse(request, "template.html", {extra_context})
# "request" is the FIRST positional arg — NOT a key inside the context dict.

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
async def login(
    request:  Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = USER_STORE.get(username)
    if not user or user["password"] != password:
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "Invalid username or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    if user["role"] == "team_leader":
        return RedirectResponse(url="/agents-dashboard", status_code=status.HTTP_303_SEE_OTHER)

    if user["role"] == "qa_admin":
        return RedirectResponse(url="/qa-supervisor", status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        request, "login.html",
        {"error": "Unable to determine user role."},
        status_code=status.HTTP_400_BAD_REQUEST,
    )


# ── Dashboard pages ───────────────────────────────────────────────────────────

@app.get("/qa-dashboard", response_class=HTMLResponse)
async def qa_dashboard_page(request: Request):
    return templates.TemplateResponse(request, "qa-dashboard.html")


@app.get("/agents-dashboard", response_class=HTMLResponse)
async def agents_dashboard_page(request: Request):
    return templates.TemplateResponse(request, "agents-dashboard.html")


@app.get("/qa-supervisor", response_class=HTMLResponse)
async def qa_supervisor_page(request: Request):
    return templates.TemplateResponse(request, "qa-supervisor.html")


# ── Agent email list ──────────────────────────────────────────────────────────

@app.get("/agents/emails")
async def list_agent_emails(
    json_file: str = Query(default="passcode.json"),
    db_key:    str = Query(default="CM"),
):
    import sys
    sys.path.insert(0, str(CM_DIR))
    from CM_DB_Handler import CMDatabaseHandler

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

    emails = df["EmailAddress"].dropna().unique().tolist()
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

    json_path = Path(json_file)
    if not json_path.is_absolute():
        json_path = CM_DIR / json_file
    if not json_path.exists():
        raise HTTPException(status_code=500, detail=f"Credentials file not found: {json_path}")

    logger.info("retrieve-and-analyze | agent=%s date=%s", agent_email, filter_date)

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

    try:
        batch = df_to_batch(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Transcript conversion failed: {exc}")

    logger.info("retrieve-and-analyze | %d calls passed validation", len(batch.calls))

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


# ── Single / batch analysis endpoints ────────────────────────────────────────

@app.post("/upload-analyze", response_model=QAAnalysisResult)
async def upload_analyze(file: UploadFile = File(...)):
    try:
        raw  = await file.read()
        data = json.loads(raw)          # json is now imported at the top
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}")

    try:
        payload = CallTranscript(**data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"JSON does not match CallTranscript schema: {exc}")

    logger.info("upload-analyze | call_id=%s agent=%s dept=%s",
                payload.call_id, payload.agent_name, payload.department)
    try:
        result = await analyzer.analyze(payload)
    except Exception as exc:
        logger.exception("upload-analyze failed | call_id=%s", payload.call_id)
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}")

    logger.info("upload-analyze done | call_id=%s assessment=%s",
                payload.call_id, result.overall_assessment)
    return result


@app.post("/analyze-call", response_model=QAAnalysisResult)
async def analyze_call(payload: CallTranscript) -> QAAnalysisResult:
    logger.info("analyze-call | call_id=%s agent=%s dept=%s duration=%ss",
                payload.call_id, payload.agent_name, payload.department,
                payload.call_duration_seconds)
    try:
        result = await analyzer.analyze(payload)
    except Exception as exc:
        logger.exception("analyze-call failed | call_id=%s", payload.call_id)
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}")

    logger.info("analyze-call done | call_id=%s assessment=%s",
                payload.call_id, result.overall_assessment)
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