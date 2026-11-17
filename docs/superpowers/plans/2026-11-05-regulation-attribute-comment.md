# Regulation Attribute and Comment Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox syntax for tracking.

**Goal:** Record the exact YAML regulation attribute and item comment for each matched QA violation, then display those fields in dashboards and the Customer Care export.

**Architecture:** Keep the existing pillar `type`, freeform `description`, transcript evidence, and assessment fields. Render exact YAML labels in focused prompts and canonicalize returned labels against a read-only catalog before building the result model. Store new labels in two nullable SQL columns; let unmatched legacy or non-catalog flags carry null labels.

**Tech Stack:** Python 3.12, Pydantic, PyYAML, SQLAlchemy, FastAPI, openpyxl, server-rendered HTML/JavaScript, SQL Server.

**Spec:** User request in this conversation, 2026-11-05.

## Global Constraints

- Do not execute database DDL or update existing rows. The user confirmed both nullable columns already exist; retain an idempotent SQL script as a reference.
- Preserve `Assessment_Reasoning`, `Flag_Description`, `Transcript_Excerpt`, existing score fields, and existing flag types.
- Preserve the one-chat-per-row Customer Care export and the four per-type error counts.
- Use exact `original_name` and `items[].violation` strings from the regulation YAML when a precise catalog match exists.
- Try catalog matches across all regulation YAML files; never invent a new policy label or rule.

## Review Focus

- Reservation YAML has a top-level attribute rather than a `pillars` key.
- Historical rows and deterministic flags may have no exact YAML label; show empty new fields.
- `Other` is an explicit catalog item, not an automatic fallback.
- Existing dashboard escalation actions still identify a row by description and excerpt.
- The user confirmed that `[Attribute]` and `[Comment]` have already been added to the QA results table.

### Task 1: Catalog and agent output

**Files:** `app/services/regulation_catalog.py`, `app/services/criteria_loader.py`, `app/prompts/qa_prompt.py`, `app/models/output.py`, `app/agent/nodes.py`, focused tests.

- [x] Write tests for exact YAML label pairs, item-ID recovery, reservation schema, unmatched flags, and preservation of assessment text.
- [x] Run tests and confirm failure for missing fields.
- [x] Add optional output fields and canonicalize only precise catalog matches during flag aggregation.
- [x] Render exact YAML labels and item IDs in regulation prompts; request `attribute` and `comment` in the focused output JSON.
- [x] Run focused tests.

### Task 2: Database contract and read API

**Files:** `app/SQL/add_regulation_attribute_comment.sql`, `app/SQL/agentDB.sql`, `app/services/sql_helpers.py`, `app/main.py`, focused tests.

- [x] Write tests for insert parameters and grouped result API fields.
- [x] Add an idempotent, nullable, add-only SQL migration script without running it.
- [x] Add the two fields to inserts and grouped API flags; leave existing fields unchanged.
- [x] Run focused tests.

### Task 3: Dashboard and Excel surfaces

**Files:** four dashboard templates, `app/services/qa_results_export.py`, focused tests.

- [x] Write a workbook test for Pillar = type, Attribute = YAML attribute, Comment = YAML item, and preserved freeform details.
- [x] Update the Customer Care exporter and show the two labels beside each flag on live and stored dashboards.
- [x] Verify JavaScript syntax and focused tests.

### Task 4: Final verification

- [x] Run `app/tests` (1663 passed; one previously identified unrelated doctor-validation test deselected) and focused tests after the final prompt update.
- [x] Check SQL script is add-only, no database was changed, and the final diff preserves assessments.
