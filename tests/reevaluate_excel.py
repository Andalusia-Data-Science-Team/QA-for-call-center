#!/usr/bin/env python3
"""Re-evaluate the November human review workbook without changing application code.

Run from the project root with venv/bin/python tests/reevaluate_excel.py --dry-run.
See tests/README_excel_reevaluation.md for live and resume commands.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from unittest.mock import patch
import uuid
import warnings
from zipfile import ZipFile

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOK = ROOT / 'AI Quality Data_November_2026.xlsx'
DEFAULT_SHEET = 'AI Quality Evaluation'
TYPES = ('C2C', 'C2B', 'C2Com', 'NC')

def normalize_phone(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    raw = str(value or '').strip()
    if not re.fullmatch(r'[+\d\s()\-]+', raw):
        raise ValueError('not a mobile number')
    digits = ''.join(str(int(c)) for c in raw if c.isdecimal())
    if digits.startswith('00966'):
        digits = digits[5:]
    elif digits.startswith('966') and len(digits) == 12:
        digits = digits[3:]
    if digits.startswith('0') and len(digits) == 10:
        digits = digits[1:]
    if not re.fullmatch(r'5\d{8}', digits):
        raise ValueError('expected a Saudi mobile number')
    return digits


def read_reviews(path: Path, sheet: str) -> list[dict]:
    # Read only: never let openpyxl rewrite unsupported Excel extensions.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*extension is not supported.*')
        book = load_workbook(path, read_only=True, data_only=False)
        try:
            ws = book[sheet]
            if [ws.cell(2, i).value for i in (2, 10, 11, 12)] != [
                'Patient Number', 'QA Evaluation', 'AI Evaluation', 'QA Validation'
            ]:
                raise ValueError('Unexpected workbook headers in B2/J2/K2/L2')
            reviews = []
            for cells in ws.iter_rows(min_row=3, max_col=15):
                values = [c.value for c in cells]
                raw_phone = values[1]
                if raw_phone is None or str(raw_phone).startswith('='):
                    continue
                phone = normalize_phone(raw_phone)
                day, clock = values[2:4]
                if not isinstance(day, datetime):
                    raise ValueError(f'Row {cells[1].row}: missing/invalid review date')
                if isinstance(clock, datetime):
                    clock = clock.time()
                if isinstance(clock, (int, float)):
                    clock = (datetime.min + timedelta(days=clock)).time()
                if not isinstance(clock, time):
                    raise ValueError(f'Row {cells[1].row}: missing/invalid review time')
                reviews.append({
                    'row': cells[1].row, 'phone': phone, 'date': day.date().isoformat(),
                    'time': clock, 'agent': str(values[4] or '').strip(),
                    'expected_counts': {k: int(values[i] or 0) for k, i in zip(TYPES, range(5, 9))},
                    'human_review': values[9], 'old_ai': values[10],
                    'old_validation': values[11], 'old_accuracy': values[13],
                })
            return reviews
        finally:
            book.close()


def select_candidate(review: dict, candidates: list[dict]) -> dict:
    unique = {str(c['Call_ID']).upper(): c for c in candidates}
    same_day = [c for c in unique.values() if str(c['Call_Date'])[:10] == review['date']]
    if not same_day:
        raise ValueError('no conversation matches the phone and review date')
    if len(same_day) == 1:
        return same_day[0]
    name = lambda s: ' '.join(str(s or '').casefold().split())
    same_agent = [c for c in same_day if name(c['Agent_Name']) == name(review['agent'])]
    pool = same_agent or same_day
    if len(pool) == 1:
        return pool[0]
    wanted = datetime.combine(datetime.fromisoformat(review['date']).date(), review['time'])
    ranked = sorted(
        ((abs((datetime.fromisoformat(str(c['Call_Date'])) - wanted).total_seconds()), i, c)
         for i, c in enumerate(pool)), key=lambda x: (x[0], x[1])
    )
    if ranked[0][0] <= 30 * 60 and ranked[1][0] - ranked[0][0] >= 5 * 60:
        return ranked[0][2]
    raise ValueError('ambiguous phone/date match; provide --conversation-map for this Excel row')


def save_manifest(path: Path, data: dict) -> None:
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(temp, path)


def lookup_candidates(db, phone: str) -> list[dict]:
    # Column and table names follow the existing stored-results dashboard.
    cursor = db.cursor()
    try:
        cursor.execute('''SELECT DISTINCT CONVERT(varchar(50), Call_ID) AS Call_ID,
                   Call_Date, Agent_Name
            FROM [DWH].[AI].[Call_QA_Results]
            WHERE RIGHT(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
                CONVERT(nvarchar(100), Customer_Phone_Number), ' ', ''), '+', ''),
                '-', ''), '(', ''), ')', ''), CHAR(9), ''), 9) = ?''', phone)
        names = [c[0] for c in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]
    finally:
        cursor.close()


def connect_results(path: Path, db_key: str):
    import pyodbc
    credentials = json.loads(path.read_text(encoding='utf-8'))['DB_NAMES']
    key = db_key or ('QA_DB' if 'QA_DB' in credentials else 'DWH')
    c = credentials[key]
    connection = pyodbc.connect(
        f"DRIVER={c['driver']};SERVER={c['Server']};DATABASE={c['Database']};"
        f"UID={c['UID']};PWD={c['PWD']};", timeout=20,
    )
    connection.timeout = 60
    return connection


def export_calls(manifest: dict, reviews: list[dict]) -> list[dict]:
    calls=[]
    for review in reviews:
        record=manifest['rows'].get(str(review['row']),{})
        result=record.get('result') or {}
        metadata=result.get('_call_metadata') or {}
        calls.append({**result, **metadata,
            'agent_name':metadata.get('agent_name') or result.get('agent_name') or review['agent'],
            'customer_phone_number':review['phone'],
            'call_date':metadata.get('call_date') or f"{review['date']}T{review['time']}",
            'call_id':result.get('call_id') or record.get('matched',{}).get('Call_ID'),
            'created_at':record.get('completed_utc'),
        })
    return calls


def write_output_sheet(manifest: dict, reviews: list[dict], details: Path) -> None:
    from io import BytesIO
    from app.services.qa_results_export import build_customer_care_workbook
    from excel_export_append import append_export
    payload=build_customer_care_workbook(export_calls(manifest,reviews))
    # Export normally assumes completed calls. Pending/error rows must not look like passes.
    book=load_workbook(BytesIO(payload))
    sheet=book.active
    for index,review in enumerate(reviews,3):
        record=manifest['rows'].get(str(review['row']),{})
        if record.get('status')!='completed':
            for col in (16,17,18,19,47,51,52,59):sheet.cell(index,col).value=None
            sheet.cell(index,50).value=record.get('error') or 'Pending re-evaluation'
    output=BytesIO();book.save(output);book.close()
    append_export(Path(manifest['workbook']),manifest['sheet'],output.getvalue(),
        {i:r['row'] for i,r in enumerate(reviews,3)},manifest['start_col'],manifest.get('excel_schema_width',0))
    manifest['excel_schema_version']=3
    manifest['excel_schema_width']=61


def fresh_rerun(previous: Path, manifest: dict) -> tuple[Path,dict]:
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=ROOT/'tests'/'reevaluation_runs'/run_id
    directory.mkdir(parents=True)
    backup=directory/Path(manifest['workbook']).name
    shutil.copy2(manifest['workbook'],backup)
    fresh={**manifest,'run_id':run_id,'rows':{},'backup':str(backup),'previous_run':str(previous.resolve())}
    fresh.pop('model',None)
    return directory/'results.json',fresh


def build_agent(run_dir: Path):
    # Settings and application imports occur only for a live run, from its root.
    from app.config import settings
    settings.NODE_CONSUMPTION_LOG_PATH = str(run_dir / 'node_consumption.log')
    settings.OVERALL_CONSUMPTION_LOG_PATH = str(run_dir / 'overall_consumption.log')
    settings.NODE_OUTPUT_LOG_PATH = str(run_dir / 'node_output.log')
    settings.TOKEN_USAGE_LOG_PATH = str(run_dir / 'token_usage.log')
    from app.agent.agent import QAAgent
    from app.services.llm_client import LLMClient
    return QAAgent(LLMClient(provider=settings.llm_provider, model=settings.llm_model))


async def evaluate_chat(agent, call_id: str, phone: str, credentials: Path):
    from CM_DB_Handler import CMDatabaseHandler
    from Chat_Retriever import retrieve_chats_by_agent_email, df_to_batch
    handler = CMDatabaseHandler(json_file=str(credentials), db_key='CM')
    df = await asyncio.to_thread(
        retrieve_chats_by_agent_email, None, None, db_handler=handler,
        sql_file=str(ROOT / 'app/SQL/NEWFIX_CM_Chat_Search.sql'),
        chat_limit=1000, conversation_id=str(uuid.UUID(call_id)),
    )
    if df is None or df.empty:
        raise ValueError('CM returned no transcript for the matched conversation ID')
    calls = df_to_batch(df).calls
    if len(calls) != 1 or calls[0].call_id.casefold() != call_id.casefold():
        raise ValueError('CM transcript does not match the requested conversation ID')
    if normalize_phone(calls[0].Patient_Phone) != phone:
        raise ValueError('CM transcript phone does not match the workbook phone')
    # Patch the symbol actually used by the compiled graph node, only in this
    # standalone process. All inference and validation nodes execute unchanged.
    with patch('app.agent.nodes.insert_qa_result', return_value={'analysis_version': None}):
        result = await agent.analyze(calls[0])
    payload=result.model_dump(mode='json')
    call=calls[0]
    payload['_call_metadata']={
        'agent_name':call.agent_name,'call_date':call.call_date,
        'archive_datetime':call.archive_datetime.isoformat() if call.archive_datetime else None,
        'call_duration_seconds':call.call_duration_seconds,'department':call.department,
        'channel_name':call.channel,'conversation_link':call.conversation_link,
        'business_unit':call.business_unit,
    }
    return payload


async def run(args) -> int:
    workbook = args.workbook.resolve()
    reviews = read_reviews(workbook, args.sheet)
    if not reviews:
        raise ValueError('No review rows found')
    print(f'Found {len(reviews)} review rows (formula footers excluded).', flush=True)
    fingerprint = hashlib.sha256(json.dumps(reviews, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()
    if args.inspect:
        return 0
    mapping = json.loads(args.conversation_map.read_text()) if args.conversation_map else {}
    if args.resume:
        details = args.resume.resolve()
        manifest = json.loads(details.read_text())
        if manifest['fingerprint'] != fingerprint or manifest['workbook'] != str(workbook) or manifest['sheet'] != args.sheet:
            raise ValueError('Resume manifest belongs to different workbook, sheet, or review data')
        if manifest['dry_run'] != args.dry_run:
            raise ValueError('Cannot resume a dry-run manifest as a live run; start a new run')
        start_col = manifest.get('start_col')
        if getattr(args, 'rerun', False):
            if args.dry_run or getattr(args,'rewrite_only',False):
                raise ValueError('--rerun cannot be combined with --dry-run or --rewrite-only')
            details,manifest=fresh_rerun(details,manifest)
    else:
        if getattr(args,'rerun',False):
            raise ValueError('--rerun requires --resume with the previous run manifest')
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_dir = ROOT / 'tests' / 'reevaluation_runs' / run_id
        run_dir.mkdir(parents=True)
        details = run_dir / 'results.json'
        manifest = {'run_id': run_id, 'workbook': str(workbook), 'sheet': args.sheet,
                    'fingerprint': fingerprint, 'dry_run': args.dry_run, 'rows': {}}
        start_col = None
    if getattr(args, 'rewrite_only', False):
        if not args.resume or args.dry_run or start_col is None:
            raise ValueError('--rewrite-only requires a live --resume manifest')
        write_output_sheet(manifest, reviews, details)
        save_manifest(details, manifest)
        print(f'Reformatted saved results in {workbook}; no database or model calls.', flush=True)
        return 0
    db = connect_results(args.credentials, args.db_key)
    try:
        agent = None
        if not args.dry_run:
            agent = build_agent(details.parent)
            current_model = f'{agent.llm_client.provider} / {agent.llm_client.model}'
            if manifest.get('model') and manifest['model'] != current_model:
                raise ValueError('Configured model changed; start a new run instead of resuming')
            manifest['model'] = current_model
            if start_col is None:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*extension is not supported.*')
                    book = load_workbook(workbook, read_only=True)
                    start_col = max(16, book[args.sheet].max_column + 1)
                    book.close()
                backup = details.parent / workbook.name
                shutil.copy2(workbook, backup)
                manifest.update(start_col=start_col, backup=str(backup), excel_schema_width=0)
            save_manifest(details, manifest)
            # Restore saved results and upgrade the earlier Excel layout without LLM calls.
            write_output_sheet(manifest, reviews, details)
        save_manifest(details, manifest)
        print(f'Run checkpoint: {details}', flush=True)
        completed = 0
        for review in reviews:
            key = str(review['row'])
            if manifest['rows'].get(key, {}).get('status') in {'completed', 'matched'}:
                continue
            if args.limit and completed >= args.limit:
                break
            record = {'status': 'error'}
            try:
                candidates = lookup_candidates(db, review['phone'])
                record['candidates'] = candidates
                if key in mapping:
                    override = str(uuid.UUID(mapping[key]))
                    matched = next((c for c in candidates if c['Call_ID'].casefold() == override.casefold()), None)
                    if matched is None:
                        raise ValueError('Mapped ID is not present among this phone\'s QA results')
                    record['match_note'] = 'Explicit row mapping supplied'
                else:
                    matched = select_candidate(review, candidates)
                    record['match_note'] = 'Matched phone/date; agent/time used when multiple IDs exist'
                record['matched'] = matched
                if ' '.join(str(matched['Agent_Name']).casefold().split()) != ' '.join(review['agent'].casefold().split()):
                    record['match_note'] += '; stored agent name differs from Excel (review manually)'
                if args.dry_run:
                    record['status'] = 'matched'
                else:
                    result = await evaluate_chat(agent, matched['Call_ID'], review['phone'], args.credentials)
                    record['result'] = result
                    if result['overall_assessment'] == 'error':
                        raise ValueError(result['assessment_reasoning'])
                    record['status'] = 'completed'
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            record['completed_utc'] = datetime.now(timezone.utc).isoformat()
            manifest['rows'][key] = record
            save_manifest(details, manifest)
            if not args.dry_run:
                write_output_sheet(manifest, reviews, details)
                save_manifest(details, manifest)
            completed += 1
            print(f"Excel row {key}: {record['status']}" + (f" — {record['error']}" if record.get('error') else ''), flush=True)
    finally:
        db.close()
    counts = Counter(r['status'] for r in manifest['rows'].values())
    print(f'Status: {dict(counts)}; untouched rows: {len(reviews) - len(manifest["rows"])}', flush=True)
    print(f'Details/resume file: {details}', flush=True)
    if not args.dry_run:
        print(f'Workbook: {workbook}\nOriginal backup: {manifest["backup"]}', flush=True)
    return 1 if counts['error'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, default=DEFAULT_BOOK)
    parser.add_argument('--sheet', default=DEFAULT_SHEET)
    parser.add_argument('--credentials', type=Path, default=ROOT / 'app/Passcode.json')
    parser.add_argument('--db-key', default='', help='QA lookup credential key (default QA_DB, otherwise DWH)')
    parser.add_argument('--dry-run', action='store_true', help='Look up IDs only; no LLM calls or Excel changes')
    parser.add_argument('--inspect', action='store_true', help='Validate/count workbook reviews without database access')
    parser.add_argument('--limit', type=int, default=0, help='Maximum additional rows this invocation (0 = all)')
    parser.add_argument('--resume', type=Path, help='Resume a results.json; completed rows are skipped')
    parser.add_argument('--rerun', action='store_true', help='With --resume: start a fresh evaluation and replace that run’s appended block (backed up first)')
    parser.add_argument('--rewrite-only', action='store_true', help='With --resume: reformat saved results only; no database/model calls')
    parser.add_argument('--conversation-map', type=Path, help='JSON object mapping Excel row numbers to explicit conversation UUIDs')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be >= 0')
    args.credentials = args.credentials.resolve()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'app/CM'))
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print('Interrupted. Use --resume with the run results.json to continue.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Run stopped: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
