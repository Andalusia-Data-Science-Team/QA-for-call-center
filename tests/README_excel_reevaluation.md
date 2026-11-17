# November human QA re-evaluation

All new code is in `tests/`. Application code and its evaluation flow are
unchanged; a process-local patch suppresses the QA-result INSERT. The workbook
contains **38 review rows (3–40)**. B119 is a formula footer, not another chat.

## Excel output

The script reuses the application's actual
`app.services.qa_results_export.build_customer_care_workbook` formatter.
Its complete Customer Care export is appended **on the same review sheet**,
one result aligned with each original review row. The existing A:O reviews
and Score sheet are preserved. The current output occupies **P:BX**.

This is the export feature's layout, with its merged section headers, colors,
column widths, numeric counts, duration formats, and clickable chat links:

| New columns | Export contents |
| --- | --- |
| P:AD | Agent/chat metadata |
| AE:AH | C2C, C2B, C2Com, NC counts |
| AI:AL | Error 1: Pillar, Attribute, Comment, Others |
| AM:AP | Error 2: Pillar, Attribute, Comment, Others |
| AQ:AT | Error 3: Pillar, Attribute, Comment, Others |
| AU:AX | Error 4: Pillar, Attribute, Comment, Others |
| AY:BB | Error 5: Pillar, Attribute, Comment, Others |
| BC:BU | Company error, monitoring, classification, coaching fields |
| BV | All Escalations, retaining every finding including errors beyond five |
| BW:BX | Chat Link, Call ID |

**Pillar** is the flag type, **Attribute** the flag name, **Comment** its
catalog comment, and **Others** contains description, severity and evidence,
exactly as in the export feature. Pending/error rows leave assessment,
classification and counts empty; Clear Comment explains their status.

The script does not invent HR IDs or other unavailable metadata, or grade the
new findings against the human review. New evaluations retain the retrieved
chat's duration, archive date, department, business unit and channel for export.

## Run

From `/home/ai/Workspace/Rafik/QA_System-main`:

```bash
# Validate/count reviews without database/model calls.
./venv/bin/python tests/reevaluate_excel.py --inspect

# Find conversation IDs without model calls or Excel changes.
./venv/bin/python tests/reevaluate_excel.py --dry-run

# Fresh evaluation, appending a new export block after existing columns.
./venv/bin/python tests/reevaluate_excel.py

# Fresh evaluation REPLACING a previous run's appended block, backed up first.
./venv/bin/python tests/reevaluate_excel.py --resume tests/reevaluation_runs/<previous-run-id>/results.json --rerun

# Continue an interrupted run; completed rows are skipped.
./venv/bin/python tests/reevaluate_excel.py --resume tests/reevaluation_runs/<current-run-id>/results.json

# Re-render saved results only, without database/model calls.
./venv/bin/python tests/reevaluate_excel.py --resume tests/reevaluation_runs/<run-id>/results.json --rewrite-only

# Offline verification.
./venv/bin/python -m unittest discover -s tests -p test_excel_reevaluation.py -v
```

Close Excel while the script writes. Only one process should write the workbook
at a time. `--limit 1` attempts one additional row (including lookup failures).
Use `--resume` afterward to continue that same run.

The existing `.env` and `app/Passcode.json` supply application credentials.
The configured provider is OpenRouter with `google/gemini-3.7-flash`; the user
approved the live evaluations through this provider. Normal API charges apply.
No web server is required. Human review notes are never included in model input.

## Monitor

The current fresh rerun writes an unbuffered log:

```bash
tail -f tests/reevaluation_runs/latest_run.log
```

To watch only row outcomes and summary lines:

```bash
tail -f tests/reevaluation_runs/latest_run.log | grep --line-buffered -E 'Run checkpoint:|Excel row|Status:|Run stopped:'
```

Ctrl+C stops watching, while the evaluation continues. Check the worker with:

```bash
pgrep -af '[p]ython.*tests/reevaluate_excel.py'
```

The log prints the new `Run checkpoint:` path near startup. Its `results.json`
records completed/error rows, matched IDs, complete model results and details.
Use that checkpoint for resume, rather than the previous run's file.

## Matching, checkpoints and preservation

Phone formats are normalized; lookup reads `[DWH].[AI].[Call_QA_Results]`
(`Customer_Phone_Number` -> `Call_ID`). Review date, agent and time resolve
multiple candidates; ambiguous IDs are never guessed. Original database lookup
matched 36 reviews. **Rows 29 and 36** had no matching phone records and are
recorded as errors. Verify their phone/original result before evaluating them.

For ambiguous candidates, `--conversation-map tests/conversation_map.json`
accepts `{"12": "00000000-0000-0000-0000-000000000001"}`. Mapped IDs must
belong to the phone in QA results; the CM phone is verified before evaluation.

Each run folder has a workbook backup and `results.json`. Fresh `--rerun`
retains the previous checkpoint/results and backs up its workbook before
replacing its generated block. Interrupted runs resume without repeating
completed chats. Resume rejects changed original review data or model settings.

Writes edit the worksheet XML and append/remap the export's styles inside the
XLSX package. Other sheets, original cells/styles, formulas, pivot data and
Excel formatting extensions remain intact. Workbook replacement is atomic;
JSON is checkpointed after every result. An interrupted export expansion is
recognized by its complete two-row export headers. Unrelated cells outside
the previous block prevent expansion instead of being overwritten.

Exit status is 1 if any attempted row failed, 0 otherwise; a limited run can
exit 0 with untouched rows remaining. Output artifacts are ignored by Git.
