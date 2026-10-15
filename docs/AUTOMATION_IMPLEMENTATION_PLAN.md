# Implementing the Automated QA Sampling Architecture in `QA_System-main`

Audience: the engineers who will build this in this repository.
Date: 14 October 2026. Baseline: working tree at `/home/ai/Workspace/Rafik/QA_System-main`.

This maps `Architecture.md` onto the code that actually exists here. Section 0 lists
what the architecture assumes that the code does not yet provide — read it first,
because two of those items are decisions that must be made before coding starts.

---

## 0. Gaps between the architecture and this codebase

Verified by reading the files, not inferred.

| # | Finding | Where | Consequence |
|---|---|---|---|
| 1 | Candidate discovery is hard-capped: `TopConversations AS (SELECT TOP (10) * ... ORDER BY Start_DateTime DESC)` | `app/SQL/NEWFIX_CM_Chat_Search.sql` | The automatic path can never see a full interval's candidates. Must be split into two queries. |
| 2 | No message deduplication in the SQL. The architecture says the pasted version had it; this file does not. | same | Duplicate `Messages` joins land in the transcript. Dedup must be added. |
| 3 | Query "parameters" are regex string substitution, not bound parameters: `re.sub(rf':\s*{key}\b', sql_value, query)` | `app/CM/CM_DB_Handler.py:166-178` | Passing a list of selected conversation IDs through this path is both injection-prone and fragile. A real bound-parameter path is needed. |
| 4 | `df_to_batch()` silently truncates to the first 50 conversations with `warnings.warn`, and `BatchCallTranscripts.calls` is `Field(..., min_length=1, max_length=50)` | `app/CM/Chat_Retriever.py:96-109`, `app/models/input.py:312` | 300/day cannot flow through `df_to_batch`. Automation must build one `CallTranscript` per job via `CallTranscript.from_cm_rows`. |
| 5 | **Persistence happens inside the graph, not around it.** `save_to_database` is LangGraph node 7, calling `insert_qa_result`; `QAAgent.analyze()` returns after it has already written. | `app/agent/nodes.py:2293-2341`, `app/agent/graph.py` | The architecture's "call the shared service around `QAAgent.analyze()` and persist with the job identity" does not fit as written. The `JobId` has to reach the node. See §4. |
| 6 | `insert_qa_result` resolves `Analysis_Version` as `MAX(Analysis_Version)+1` per `Call_ID` | `app/services/sql_helpers.py:236-247` | Every retry creates a new version row. Not idempotent, and with no DDL rights on DWH it cannot be fixed with a column — see §2.2 for the protocol that replaces it. |
| 7 | Timezone is a hard-coded `DATEADD(HOUR, 3, ...)` on every timestamp, and the date filter applies `DATEADD(HOUR, -3, ...)` | `app/SQL/NEWFIX_CM_Chat_Search.sql` | Fixed +03:00. Correct for `Asia/Riyadh` year-round; **wrong for `Africa/Cairo` in winter** (Egypt is UTC+2 outside DST). Reporting-day boundaries will be off by an hour for Cairo for part of the year. |
| 8 | Attribution uses `c.InboxOwnerPersonUniqueId` (current inbox owner) only | same | This is exactly the attribution the architecture says is insufficient for transferred chats. The data to do better exists: `Messages.Sender_UniqueId → Participations → People`, plus `Discriminator = 'ConversationForwardedSystemMessage'`. |
| 9 | No roster shift source. `DWH.AI.Agents` exposes `Agent_id, Agent_Name, Department, is_Active, Created_at, Agent_Email_Address, Agent_Email_Password, Level, Supervisor_Name` | `app/SQL/DWH_Agents.sql` | **Resolved by dropping the shift model.** Sampling covers every agent across the full reporting day. See §7. |
| 10 | `_build_engine()` builds a brand-new SQLAlchemy engine on every call | `app/services/sql_helpers.py:53-71` | Acceptable for a manual request; a worker loop doing this per job will churn connection pools. Make it module-level and cached. |
| 11 | No scheduler dependency (no APScheduler in `requirements.txt`), no `Dockerfile`, no compose file | `requirements.txt`, repo root | The automation container in §3 of the architecture has nothing to run in. |
| 12 | `from_cm_rows` sorts messages by `CreationDateTime` only, with no message-ID tiebreak and no dedup | `app/models/input.py:209-220` | Ordering is non-deterministic for same-timestamp messages. |
| 13 | Evaluation is sequential by explicit choice; the `asyncio.gather` path is commented out | `app/main.py:325-335` | Fine as the starting point — matches `QA_WORKER_CONCURRENCY=1` — but throughput for 300/day must be measured before scaling. |

### Two decisions needed before coding

**D1 — Where do agent working periods come from?** Without them, per-agent interval
allocation is undefined. Three options, in order of preference:

- **(a)** A real shift/roster table in `DWH.AI` (new table, fed by WFM or HR).
  Correct, and matches the architecture as written.
- **(b)** Derive the working period per agent per day from observed activity:
  first to last attributable handled message, rounded out to interval boundaries.
  Cheap, needs no new feed, but it is circular — an agent with no traffic gets no
  quota, and the day's shape is only knowable after the day ends, which breaks the
  "freeze the roster before the first interval is due" requirement.
- **(c)** A static shift pattern per agent in config, keyed by `Agent_Email_Address`.
  Workable stopgap; goes stale silently.

Recommendation: **(a)**, with **(c)** as the launch fallback so phase 3 is not blocked.
Do not build on (b) — it cannot satisfy §5's freeze-before-the-day requirement.

**D2 — Attribution rule for transferred chats.** The architecture requires a
"responsible human agent" and explicitly rejects inbox-owner-alone. Proposed concrete
rule, to be confirmed against real transferred examples:

> The responsible agent is the human who sent the **last** non-system message
> in the conversation, resolved via `Messages.Sender_UniqueId → Participations
> → People` with `People.Discriminator` identifying a human agent. Their final
> message timestamp is the handling timestamp used for day and interval
> assignment. If no human message exists, the conversation is ineligible. If
> the resolved identity is ambiguous, quarantine with a reason.

Both decisions belong in stage 1 of the delivery plan and should be signed off by the
QA business owner, not chosen in code review.

---

## 1. Target module layout

All new code under the existing `app/` package, no new top-level directories.

```
app/
  automation/
    __init__.py
    config.py          # AutomationSettings (the QA_* env block)
    calendar.py        # reporting-day / interval math, tz-aware
    roster.py          # agent roster + working periods (D1)
    sampling.py        # deterministic hash ranking + nonconsecutive selection
    db.py              # SQLite ledger: connect, pragmas, schema, fencing
    planner.py         # daily plan: roster freeze, quotas, interval rows
    dispatcher.py      # 10-min loop: find due intervals, snapshot, enqueue
    worker.py          # claim job → retrieve → evaluate → persist
    reconciler.py      # end-of-day deficit fill and recovery
    repository.py      # all SQL against the QA_* ledger tables
    runner.py          # process entrypoint: python -m app.automation.runner
  services/
    evaluation_service.py   # shared wrapper over QAAgent.analyze (§4)
  CM/
    candidate_queries.py    # bound-parameter CM_DWH access (§3)
  SQL/
    automation/
      sqlite/
        001_ledger.sql          # the local ledger (the only DDL we own)
      mssql_reference/          # NOT built — needs DDL rights we do not have
      candidates_by_interval.sql   # read-only against CM_DWH
      transcripts_by_ids.sql       # read-only against CM_DWH
```

---

## 2. Where state lives: local SQLite, not DWH

**We have no DDL rights on DWH.** That is a hard constraint, and it rules out
the original plan for this section, which added tables to `DWH.AI` and a
`QA_Job_Id` column to `DWH.AI.Call_QA_Results`. Neither is possible.

The division is therefore:

| Store | Access | Contents |
|---|---|---|
| **DWH / CM_DWH** (SQL Server) | **read + insert only — never DDL** | Conversations, messages, `DWH.AI.Agents`, and `DWH.AI.Call_QA_Results`, which automation inserts into through the existing `insert_qa_result` path, unchanged |
| **Local SQLite** | owned outright | Runs, quotas, intervals, candidate snapshots, jobs, leases, result links |

Schema: [`app/SQL/automation/sqlite/001_ledger.sql`](../app/SQL/automation/sqlite/001_ledger.sql).
Connection and bootstrap: [`app/automation/db.py`](../app/automation/db.py).
Applied automatically on first use; idempotent, so it runs on every startup.

The T-SQL originals are kept under `app/SQL/automation/mssql_reference/` and
are **not** part of the build — see the README there. They matter only if DDL
rights on DWH are granted later.

### 2.1 Ledger tables

`QA_SamplingRuns`, `QA_AgentQuotas`, `QA_SamplingIntervals`,
`QA_SamplingCandidates`, `QA_EvaluationJobs`, `QA_ResultLinks`, plus a
`QA_FenceCounter` and a `QA_RunCoverage` view. SQLite-specific choices:

- **STRICT tables**, so a mistyped value raises instead of being silently
  coerced — SQLite would otherwise store `'three hundred'` in an INTEGER.
- **Timestamps as TEXT**, ISO-8601 UTC `YYYY-MM-DDTHH:MM:SS.sssZ`. That format
  sorts lexicographically in chronological order, which is what lets the
  dispatcher's `WHERE SnapshotCutoffUtc <= ?` work on plain text. Never a
  local time; `app/automation/db.py:utc_text` rejects naive datetimes.
- **Generated STORED columns** for `DayHours`, `WorkMinutes` and
  `IntervalMinutes`, computed via `julianday`. A test asserts `IntervalMinutes`
  agrees with `app.automation.calendar`, since quota is allocated
  proportionally to duration.
- **Partial unique indexes** carry the guarantees, exactly as the T-SQL design
  intended. `UX_QA_Jobs_Conv_Policy` on `(ConversationId, SamplingPolicyVersion)
  WHERE Status <> 'superseded'` is what makes the daily budget hold — the
  index, not application logic.
- **WAL + `busy_timeout` + `synchronous=FULL`**, set in `db.py`. FULL is
  deliberate: the persistence protocol below depends on a row being durable
  before an external insert, and WAL's default NORMAL can lose the most recent
  commits on an OS crash.

SQLite allows one writer at a time. At `QA_WORKER_CONCURRENCY=1` that is a
non-issue; it is the thing to revisit if concurrency ever goes well beyond a
handful of workers.

**Back the file up.** Frozen candidate snapshots are the audit trail for past
selections and cannot be reconstructed once the source data has moved on.

### 2.2 Idempotent persistence without a DWH column

The original design let a unique index on `QA_Job_Id` reject a retry's
duplicate insert. Without that column, the guarantee has to come from a
protocol instead. `QA_ResultLinks` implements it:

1. **Before inference**, look for a `linked` row for this job. Present means
   the result is already in DWH: mark the job completed, do not call the LLM.
2. **Otherwise write an `attempting` row and commit it *before* the DWH
   insert**, recording `PreInsertMaxVersion` (today's `MAX(Analysis_Version)`
   for this `Call_ID`) and `AttemptStartedUtc`. Committing first is the whole
   point — a crash after the insert must still leave evidence.
3. **After `insert_qa_result` returns**, update the row to `linked` with the
   version and row count that landed.

Recovery, when a retry finds an `attempting` row: re-read DWH for rows with
this `Call_ID` where `Analysis_Version > PreInsertMaxVersion` AND
`Created_at >= AttemptStartedUtc`. Found means the insert landed before the
crash — adopt the version, mark `linked`, do not re-infer. Not found means it
is safe to retry in full.

**Residual risk, stated plainly.** The `Created_at` window narrows but does not
eliminate the case where a *manual* re-analysis of the same `Call_ID` lands
inside the window and is mistaken for ours. `Call_QA_Results` has no column we
may add to disambiguate, and `Created_at` has one-second resolution. The window
is one crashed attempt long, and the collision requires a human to re-analyse
the exact conversation an automated job is mid-flight on. A run that hit it
would under-count by one rather than duplicate. This is strictly weaker than
the unique index would have been, and it is the price of having no DDL rights.

## 3. Splitting candidate discovery from transcript retrieval (gaps 1-3)

Two new SQL files, neither of which goes through the regex substitution path.

**`candidates_by_interval.sql`** — lightweight, one row per conversation, no
`Messages` content join, no `TOP`. Inputs are the UTC interval bounds (already
converted by Python, per §4 of the architecture) and the snapshot cutoff. It returns
`ConversationId`, the attributed agent (per D2), the handling timestamp, the
conversation state and the archive time. Candidate discovery is a metadata query —
it must stay cheap, because it runs every 10 minutes across every due interval.

**`transcripts_by_ids.sql`** — full message join for an explicit ID list, with the
dedup gap 2 requires:

```sql
;WITH Deduped AS (
    SELECT m.*, ROW_NUMBER() OVER (
        PARTITION BY m.Conversation_UniqueId, m.UniqueId
        ORDER BY m.CreationDateTime, m.UniqueId
    ) AS rn
    FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m
    WHERE m.Conversation_UniqueId IN (SELECT value FROM @Ids)
)
SELECT ... FROM Deduped WHERE rn = 1 ...
ORDER BY Conversation_UniqueId, CreationDateTime, MessageId;
```

The ID list is passed as a table-valued parameter or via
`STRING_SPLIT(?, ',')` with the joined IDs as a single bound `?`. Add to
`CM_DB_Handler` (or to the new `candidate_queries.py`, which is cleaner — it leaves
the existing handler untouched for the manual path):

```python
def execute_query_params(self, query_file: str, params: tuple) -> pd.DataFrame:
    """Bound-parameter execution. Unlike execute_query_from_file, the SQL text
    is never rewritten — values are passed to the driver as ? parameters."""
```

Do not extend the regex substitution to handle lists.

Timezone (gap 7): remove the `DATEADD(HOUR, 3, ...)` literals from the new automation
queries. Select raw UTC, pass tz-aware UTC bounds in as parameters computed in
`app/automation/calendar.py` with `zoneinfo`, and convert to local only for display.
Leave the existing `NEWFIX_CM_Chat_Search.sql` alone so the manual dashboard route is
not disturbed.

---

## 4. The shared evaluation service, and the job-identity problem (gaps 5, 6)

The architecture's §9.8 assumes persistence sits outside `analyze()`. Here it is graph
node 7. Three ways out; pick the first.

**Option A (recommended) — thread the job id through `AgentState`.**

1. Add `qa_job_id: Optional[str]` to `AgentState` in `app/agent/state.py`.
2. Add an optional `job_id` parameter to `QAAgent.analyze(call, *, job_id=None)` and
   seed it into `initial_state`.
3. In `save_to_database`, read `state.get("qa_job_id")` and pass it to
   `insert_qa_result(..., qa_job_id=...)`.
4. In `insert_qa_result`, when `qa_job_id` is not None, wrap the insert in the
   `QA_ResultLinks` protocol from §2.2: commit the `attempting` row first,
   insert, then record the version that landed. The `MAX+1` version logic stays
   exactly as it is — we cannot change the DWH table, so the local link is what
   makes a retry recognise work it already did.

This is roughly 30 lines across four files, keeps one persistence path, and leaves
every existing caller working because both new parameters default to `None`.

Option B (move persistence out of the graph into the service) is cleaner in the
abstract but changes manual behaviour and touches the compiled graph — not worth it
now. Option C (a separate automated results table) duplicates the schema and splits
the dashboard's source of truth. Avoid both.

`app/services/evaluation_service.py` then becomes thin:

```python
async def evaluate_conversation(call: CallTranscript, *, job_id: str) -> QAAnalysisResult:
    """Single entry point used by both the manual route and the worker."""
```

Also note `save_to_database` swallows every exception as non-fatal
(`app/agent/nodes.py:2334-2340`). For an automated job that is wrong — a job whose
result was never persisted must not be marked `completed`. The node should record the
failure into state so the worker can see it and move the job to `retry_wait`.

---

## 5. Sampling (architecture §6), concretely

`app/automation/sampling.py`, pure functions, no I/O — this is the part that must be
unit-testable without a database.

```python
def rank_hash(base_seed, policy_version, reporting_date, shift_id,
              agent_id, interval_id, conversation_id) -> str:
    canonical = "|".join([
        base_seed, policy_version, reporting_date.isoformat(), shift_id,
        str(agent_id), interval_id, str(conversation_id).lower(),
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

Canonical serialization is fixed here and nowhere else: UTF-8, ISO dates, lowercased
UUIDs, `|` separator, this field order. Never `hash()` — it is salted per process and
would silently break reproducibility across restarts.

Selection: sort candidates by `(rank_hash, conversation_id)`, walk in order, accept a
candidate only if `SequencePosition ± 1` is not already selected for that agent and
shift — checked against persisted selections so it holds across interval and midnight
boundaries. Stop at quota. Return both the selection and the shortfall with its cause;
per §6, report the greedy policy's outcome rather than claiming infeasibility.

Allocation: `floor(300/A)` per agent with a date-seeded deterministic ranking for the
`300 mod A` remainder; largest-remainder by interval duration within each agent.

---

## 6. Loops and entrypoint (gap 11)

Add `apscheduler==3.10.4` to `requirements.txt`, or — simpler and with one less
dependency — run two `asyncio` tasks on their own sleep cycles inside
`app/automation/runner.py`. Either way, **do not hook this into FastAPI's `lifespan`**
(`app/main.py:38`): multiple uvicorn workers would each start a scheduler.

```
python -m app.automation.runner        # planner + dispatcher + worker, one process
```

Guard the whole thing on `QA_AUTOMATION_ENABLED=false` by default, so the module can
be merged and deployed long before it is switched on.

Job claiming must be atomic. SQLite has no `UPDATE ... OUTPUT` and no
`READPAST`, so the claim is a read-modify-write inside `BEGIN IMMEDIATE`,
which takes the write lock up front rather than at the first write:

```python
with transaction(ledger):                      # BEGIN IMMEDIATE
    row = ledger.execute("""
        SELECT JobId FROM QA_EvaluationJobs
         WHERE Status IN ('pending', 'retry_wait')
           AND (NextAttemptUtc IS NULL OR NextAttemptUtc <= ?)
         ORDER BY CreatedAtUtc
         LIMIT 1
    """, (utc_text(),)).fetchone()
    if row:
        ledger.execute("""
            UPDATE QA_EvaluationJobs
               SET Status = 'running', LeaseOwner = ?, LeaseToken = ?,
                   LeaseExpiresUtc = ?, Attempts = Attempts + 1, FenceToken = ?
             WHERE JobId = ?
        """, (owner, str(uuid4()), utc_text(expiry), next_fence_token(ledger), row["JobId"]))
```

`BEGIN IMMEDIATE` is what makes this safe: SQLite's default deferred
transaction takes the lock only at the first write, so two workers could both
read the same row and one would then fail on upgrade. Taking the lock
immediately turns that race into a short wait governed by `busy_timeout`.
`app/automation/db.py:transaction` defaults to it for this reason.

---

## 7. Delivery status

| Stage | Scope | Status |
|---|---|---|
| **1. Decisions** | D1 roster source, D2 attribution rule, completion states | **OPEN** — implemented behind interfaces under stated assumptions; see below |
| **2. Foundations** | SQLite ledger, `db.py`, `config.py`, `calendar.py` | **Done** |
| **3. Retrieval split** | `day_candidates.sql`, `transcripts_by_ids.sql`, `candidate_queries.py` | **Done** |
| **4. Deterministic selection** | `sampling.py`, `roster.py`, `planner.py`, `dispatcher.py`, `eligibility.py` | **Done** |
| **5. Evaluation path** | `qa_job_id` threading, `evaluation_service.py`, `worker.py`, `QA_ResultLinks` | **Done** |
| **6. Reconciliation** | `reconciler.py`, `runner.py`, coverage reporting | **Done** — go-live still gated on §8 |

162 automation tests pass. The pre-existing suite is unchanged by this work:
`app/tests/test_graph_routing.py` hangs and four tests in `tests/` fail, both
verified identical on a tree with these changes stashed.

### Decisions

**D1 — working periods: cancelled, not deferred.**

The shift model has been removed. There is no `ShiftSource`, no shift config
file, and no `QA_SHIFT_*` setting. Every active agent in `DWH.AI.Agents` is
sampled across the **whole reporting day**, split into
`QA_SAMPLING_INTERVAL_MINUTES` slices so coverage spans the full 24 hours.

The model was removed because it never produced more than one answer. There
is no shift feed to read and none can be created, so every agent fell through
to a whole-day default anyway. What remained was an interface with three
implementations, a JSON config, a provenance column and an unresolvable
business question — all of it resolving to "the whole day". Deleting it
removed a schema column, two settings, a file and roughly 150 lines, and
changed no behaviour.

If a real shift feed ever appears, `plan_day` is the only caller that needs
an agent's time bounds, and it is the place to reintroduce them.

**D2 — attribution: still open, implemented as proposed.**

The responsible agent is the human who sent the **last non-system message**,
resolved `Messages.Sender_UniqueId → Participations → People` with
`People.Discriminator = 'User'`. Their final message timestamp decides the
reporting day and interval. The current inbox owner is deliberately *not*
used, but is returned as `InboxOwnerEmail` with an `OwnerDiffersFromHandler`
flag so the two can be compared on real transferred chats before sign-off.
Every candidate row stores `AttributionRule = 'last_human_message'`.

**Completion state.** `ArchiveDateTime IS NOT NULL` is the only completion
signal used. CM's `State` and `AnswerState` codes are carried through but
never used as rules, because their meanings are unconfirmed and inventing a
status code is how a sampler silently drops half a day.

### Reaching the daily target

Per-interval quotas are even; real traffic is not. An interval allotted two
conversations that saw one loses the other slot, and no later interval can
use it. Measured at `target=300` on simulated uneven traffic, that cost 5% of
the target on a busy day and more than half on a thin one.

`app/automation/backfill.py` implements the redistribution architecture §7
calls for: round 1 tops each agent up to their own quota from any interval of
the day, round 2 spreads the remainder across agents in the date-seeded
deterministic order. It works from the already-frozen candidate snapshots, so
it costs no extra CM_DWH query.

| scenario (target 300) | without backfill | with |
|---|---|---|
| 20 agents x 60 chats | 284 | **300** |
| 30 agents x 30 chats | 243 | **300** |
| 20 agents, uneven 5-80 | 227 | **300** |
| 20 agents x 60, 70% closed | 275 | **300** |
| 25 agents x 12 (thin) | 135 | 142 |

It relaxes nothing. Spacing and eligibility are never loosened to hit the
number, and a genuinely thin day closes short with the deficit recorded —
a test asserts exactly that. The remaining gaps above are real data limits:
25 agents x 12 conversations under a no-adjacent rule caps out near 150.

## 7a. Running it

```bash
python -m app.automation.runner dry-run --date 2026-10-29   # plan + select, NO LLM calls
python -m app.automation.runner status  --date 2026-10-29
python -m app.automation.runner plan    --date 2026-10-29
python -m app.automation.runner dispatch
python -m app.automation.runner work
python -m app.automation.runner reconcile --date 2026-10-29
python -m app.automation.runner backfill  --date 2026-10-29  # top up a short day
python -m app.automation.runner repair                       # unstick a wedged ledger
python -m app.automation.runner                              # the daemon
```

`dry-run` is the gate before go-live. It plans and dispatches a whole day,
reserves the jobs, and stops — the worker never starts, so not one token is
spent. It prints a `selection_fingerprint`; run it twice against fresh
ledgers and compare, which is the determinism acceptance criterion made
executable.

The daemon refuses to start unless `QA_AUTOMATION_ENABLED=true`, and runs the
planner, dispatcher and worker as independent asyncio tasks so a slow
evaluation cannot delay dispatch. It is **not** wired into FastAPI's lifespan:
uvicorn with several workers would start one scheduler per worker, and three
schedulers planning the same day is how a budget of 300 becomes 900.

## 8. Cost and throughput, before stage 6

300 conversations/day through a LangGraph pipeline that makes several LLM calls per
conversation is the operating cost of this system, and nothing in the current code
measures it at that volume. The telemetry already exists —
`logs/overall_consumption.log` records per-call `total_tokens` and `cost_usd` via
`QAAgent.analyze` — so before enabling automation, run a 20-30 conversation pilot
through the worker and multiply out. At `QA_WORKER_CONCURRENCY=1` and sequential
evaluation, also confirm that 300 jobs fit inside a day's wall clock with margin; if
they do not, raising concurrency is a change to make deliberately with measurements,
not at go-live.
