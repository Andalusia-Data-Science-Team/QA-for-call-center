# ── TEMPORARY: direct-execution import bootstrap ────────────────────────────
# Only takes effect when this file is run directly (`python main.py` from
# inside app/, or `python app/main.py`) — in that case Python puts this
# file's own directory on sys.path[0] instead of the project root, so the
# `from app.xxx import yyy` absolute imports below would fail. `__package__`
# is None/"" only for direct script execution — it's "app" when loaded
# normally via `uvicorn app.main:app` or `python -m app.main`, so this is a
# no-op (and therefore safe) for every other way of running the app.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json                          # ← was missing; needed by upload_analyze
import time
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query, Form, status
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
# StaticFiles removed — it was imported but never used
from fastapi.templating import Jinja2Templates

from app.models.input import CallTranscript, BatchCallTranscripts
from app.models.output import QAAnalysisResult, BatchQAAnalysisResult
from app.agent import QAAgent
from app.services.llm_client import LLMClient
from app.services.sql_helpers import DatabaseWritePermissionError, update_escalation_row
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

@app.post("/escalations")
async def create_escalation(payload: dict):
    call_id = payload.get("call_id")
    flag_description = payload.get("flag_description")
    transcript_excerpt = payload.get("transcript_excerpt")
    supervisor_name = payload.get("supervisor_name")
    supervisor_email_address = payload.get("supervisor_email_address")
    conversation_link = payload.get("conversation_link")
    if not call_id:
        raise HTTPException(status_code=400, detail="call_id is required")

    try:
        updated = update_escalation_row(
            call_id,
            escalated=True,
            qa_reviewed=False,
            qa_review_comment=None,
            supervisor_name=supervisor_name,
            supervisor_email_address=supervisor_email_address,
            coaching_status="Pending Review",
            flag_description=flag_description,
            transcript_excerpt=transcript_excerpt,
            conversation_link=conversation_link,
        )
    except DatabaseWritePermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if updated == 0:
        raise HTTPException(status_code=404, detail="No QA result row found to update")
    return {"ok": True, "updated": updated}


@app.post("/escalations/review")
async def review_escalation(payload: dict):
    call_id = payload.get("call_id")
    review_comment = payload.get("review_comment")
    if not call_id:
        raise HTTPException(status_code=400, detail="call_id is required")

    try:
        updated = update_escalation_row(
            call_id,
            qa_reviewed=True,
            qa_review_comment=review_comment if review_comment is not None else "",
            coaching_status="Reviewed",
        )
    except DatabaseWritePermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if updated == 0:
        raise HTTPException(status_code=404, detail="No QA result row found to update")
    return {"ok": True, "updated": updated}


@app.post("/escalations/dismiss")
async def dismiss_escalation(payload: dict):
    call_id = payload.get("call_id")
    if not call_id:
        raise HTTPException(status_code=400, detail="call_id is required")

    try:
        updated = update_escalation_row(
            call_id,
            qa_reviewed=False,
            qa_review_comment=None,
            coaching_status="Dismissed",
        )
    except DatabaseWritePermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if updated == 0:
        raise HTTPException(status_code=404, detail="No QA result row found to update")
    return {"ok": True, "updated": updated}


@app.get("/agents/emails")
async def list_agent_emails(
    json_file: str = Query(default="/home/ai/Workspace/Rafik/QA_System-main/app/Passcode.json"),
    db_key:    str = Query(default="DWH"),
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

    sql_file = str(SQL_DIR / "DWH_Agents.sql")
    try:
        df = handler.execute_query_from_file(sql_file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
    finally:
        handler.close()

    emails = df["Agent_Email_Address"].dropna().unique().tolist()
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
            return QAAnalysisResult.error_result(call.call_id, str(exc), conversation_link=call.conversation_link)

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


# ── Upload-and-test convenience helpers ──────────────────────────────────────
#
# /upload-analyze (below) requires the uploaded JSON to already satisfy the
# full CallTranscript schema. That's correct for production use, but makes it
# awkward to quickly try a hand-written or partially-extracted transcript
# (e.g. missing Patient_Phone, or call_date/department left null) without
# editing the file to add boilerplate that has nothing to do with what's
# actually being tested. _fill_test_defaults() patches ONLY the in-memory
# dict used for this one request — it never writes back to the uploaded file.

TEST_DEFAULT_DEPARTMENT = "Helpdesk"   # a generic bucket valid under VALID_DEPARTMENTS
TEST_DEFAULT_PHONE = "500000000"       # placeholder — passes the 9–15 digit validator


def _fill_test_defaults(data: dict) -> tuple[dict, list[str]]:
    """
    Return a copy of `data` with missing/null required CallTranscript fields
    replaced by clearly-fake placeholder values, plus the list of field names
    that were defaulted (so the caller can log/report exactly what was faked).

    `transcript` is intentionally NOT defaulted — it's the one field a test
    can't meaningfully fake, so a missing/blank transcript still fails
    validation with a normal 422.
    """
    patched = dict(data)
    defaulted: list[str] = []

    def _default(key: str, value):
        if patched.get(key) in (None, ""):
            patched[key] = value
            defaulted.append(key)

    _default("call_id", f"TEST-{uuid.uuid4().hex[:8].upper()}")
    _default("agent_name", "Test Agent")
    _default("Patient_Phone", TEST_DEFAULT_PHONE)
    _default("call_date", date.today().isoformat())
    _default("call_duration_seconds", 0)
    _default("department", TEST_DEFAULT_DEPARTMENT)

    return patched, defaulted


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


@app.post("/upload-analyze-test", response_model=QAAnalysisResult)
async def upload_analyze_test(file: UploadFile = File(...)):
    """
    Dev/testing convenience: upload ANY transcript-shaped JSON file — even one
    with missing or null fields (call_date, call_duration_seconds, department,
    Patient_Phone) — and run it through the same analysis pipeline as
    /upload-analyze. Missing/null fields are filled with obvious placeholder
    values (see _fill_test_defaults) for THIS request only; the uploaded file
    itself is never modified. `transcript` is still required — it's the one
    field a real test needs actual content for.

    NOT intended for production data entry — use /upload-analyze or
    /analyze-call for that, where the caller is expected to supply the real,
    complete CallTranscript fields.
    """
    try:
        raw  = await file.read()
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}")

    patched, defaulted = _fill_test_defaults(data)
    if defaulted:
        logger.info(
            "upload-analyze-test | filename=%s | defaulted fields (placeholder values, "
            "not from the file): %s",
            file.filename, defaulted,
        )

    try:
        payload = CallTranscript(**patched)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"JSON does not match CallTranscript schema even after filling test defaults {defaulted}: {exc}",
        )

    logger.info("upload-analyze-test | call_id=%s agent=%s dept=%s (defaulted=%s)",
                payload.call_id, payload.agent_name, payload.department, defaulted)
    try:
        result = await analyzer.analyze(payload)
    except Exception as exc:
        logger.exception("upload-analyze-test failed | call_id=%s", payload.call_id)
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}")

    logger.info("upload-analyze-test done | call_id=%s assessment=%s",
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


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORARY / DIRECT-EXECUTION TESTING BLOCK
# ─────────────────────────────────────────────────────────────────────────────
# Lets you run this file directly to push one local JSON transcript through
# the EXACT SAME validation + analysis path /upload-analyze uses — no
# FastAPI server, no HTTP call, no uvicorn — so you can quickly exercise the
# Location and Bank Account features (both in app/service_hub/) —
# two fully independent nodes, each with its own graph node, state key, and
# output field.
#
# It deliberately does NOT re-implement any analysis logic: it calls the
# same `CallTranscript` schema and the same module-level `analyzer`
# (QAAgent) instance that every HTTP endpoint above already uses.
#
# Change which file gets analyzed by editing TEST_JSON_PATH below. A
# relative path is resolved against this file's own directory (app/), so it
# works the same regardless of which of the two run commands you use.
# ─────────────────────────────────────────────────────────────────────────────

# TEST_JSON_PATH = "chats/fayrouz_ahmed_loc.json"   # ← change this to test a different file


# def _resolve_test_json_path(raw_path: str) -> Path:
#     """Relative paths are resolved against app/ (this file's directory), not cwd."""
#     p = Path(raw_path)
#     return p if p.is_absolute() else (Path(__file__).parent / p)


# async def _run_local_test(json_path: Path) -> None:
#     """Direct-execution counterpart of /upload-analyze — same validation,
#     same analyzer, same result type — just reading from disk and printing
#     to the terminal instead of going through FastAPI."""
#     print(f"\n{'=' * 70}\nLoading test transcript: {json_path}\n{'=' * 70}")

#     if not json_path.exists():
#         print(f"ERROR: file not found: {json_path}")
#         print(f"       (edit TEST_JSON_PATH near the bottom of {__file__} to point at your file)")
#         return

#     try:
#         data = json.loads(json_path.read_text(encoding="utf-8"))
#     except (json.JSONDecodeError, UnicodeDecodeError) as exc:
#         print(f"ERROR: invalid JSON file: {exc}")
#         return

#     # Same schema validation /upload-analyze performs — via the same
#     # CallTranscript model, not a re-implementation of it.
#     try:
#         payload = CallTranscript(**data)
#     except Exception as exc:
#         print(f"ERROR: JSON does not match CallTranscript schema: {exc}")
#         return

#     print(f"call_id       : {payload.call_id}")
#     print(f"agent_name    : {payload.agent_name}")
#     print(f"department    : {payload.department}")
#     print(f"business_unit : {payload.business_unit}")

#     # Cheap, read-only preview of the two independent deterministic gates —
#     # reuses the real detectors from app.service_hub (two separate
#     # features/nodes, not a combined one) so you can confirm
#     # detection is firing correctly even before a full, LLM-backed analysis
#     # completes.
#     from app.service_hub.bank_validation import detect_bank_signals, is_supported_bank_bu
#     from app.service_hub.location_validation import detect_location_signals
#     bank_requested, supplied, _patient_text, _agent_text, resolved_bu = detect_bank_signals(payload)
#     location_requested, _loc_patient_text, _loc_agent_text = detect_location_signals(payload)
#     print(f"bank request detected in patient text     : {bank_requested}")
#     print(f"resolved business unit                    : {resolved_bu} (in bank scope: {is_supported_bank_bu(resolved_bu)})")
#     print(f"location request detected in patient text : {location_requested}")
#     print(f"financial identifiers found in agent text : {len(supplied)}")

#     print("\nRunning full analysis via the same QAAgent instance /upload-analyze uses...\n")
#     try:
#         result = await analyzer.analyze(payload)   # identical call to every endpoint above
#     except Exception as exc:
#         print(f"ERROR: analysis failed: {exc}")
#         return

#     print(f"{'=' * 70}\nRESULT\n{'=' * 70}")
#     print(result.model_dump_json(indent=2))

#     print(f"\noverall_assessment : {result.overall_assessment}")
#     # Bank and location are independent nodes/fields now — printed separately.
#     if result.bank_validation:
#         print("bank_validation (app.service_hub):")
#         print(json.dumps(result.bank_validation, indent=2, ensure_ascii=False))
#     else:
#         print("bank_validation : None (gate did not detect a bank request in this transcript)")
#     if result.location_validation:
#         print("location_validation (app.service_hub):")
#         print(json.dumps(result.location_validation, indent=2, ensure_ascii=False))
#     else:
#         print("location_validation : None (gate did not detect a location request in this transcript)")


# if __name__ == "__main__":
#     import sys as _sys

#     if "--serve" in _sys.argv:
#         # Preserves the previous "start the real server" behaviour, now opt-in
#         # via `python main.py --serve`, since the default direct-execution
#         # behaviour is now the local JSON test above.
#         # `reload=True` is intentionally NOT set: uvicorn's auto-reload only
#         # works when the app is passed as an import string ("app.main:app"),
#         # not an already-instantiated object. Use
#         # `uvicorn app.main:app --reload` from the CLI for that.
#         import os
#         import uvicorn

#         port = int(os.getenv("PORT", "8000"))
#         uvicorn.run(app, host="127.0.0.1", port=port)
#     else:
#         import asyncio

#         asyncio.run(_run_local_test(_resolve_test_json_path(TEST_JSON_PATH)))

