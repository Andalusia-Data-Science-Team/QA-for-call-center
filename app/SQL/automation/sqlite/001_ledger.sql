/* ============================================================================
   Automated QA sampling ledger — LOCAL SQLite schema
   Requires SQLite 3.31+ (generated columns); developed against 3.45.

   WHY SQLITE
   ──────────
   We have no DDL rights on DWH. The entire ledger therefore lives in a local
   SQLite file. DWH is treated as strictly read-plus-insert:

       READ   : CM_DWH conversations and messages, DWH.AI.Agents,
                DWH.AI.Call_QA_Results (for reconciliation)
       INSERT : DWH.AI.Call_QA_Results, through the existing
                app/services/sql_helpers.py:insert_qa_result path, unchanged
       NEVER  : CREATE, ALTER, DROP, or any other schema change

   CONVENTIONS
   ───────────
   * All timestamps are TEXT, ISO-8601 UTC, 'YYYY-MM-DDTHH:MM:SS.sssZ'.
     That format sorts lexicographically in chronological order, which is what
     lets plain '<' and 'ORDER BY' work on them. Never store a local time.
   * All ids are TEXT (lowercase UUID), because SQLite has no uuid type and
     the conversation ids come from SQL Server as uniqueidentifier strings.
   * STRICT tables, so a mistyped value is an error rather than a silent
     coercion — SQLite would otherwise happily store 'banana' in an INTEGER.
   * Idempotent: every statement is IF NOT EXISTS. Safe to re-run on an
     existing ledger.

   CONCURRENCY
   ───────────
   SQLite allows one writer at a time. With QA_WORKER_CONCURRENCY=1 (the
   default, and what the architecture starts from) that is a non-issue. WAL
   mode plus busy_timeout, both set in app/automation/db.py, let readers run
   during a write and make brief contention wait rather than fail. If
   concurrency is ever raised well beyond a handful of workers, this file is
   the thing to revisit — not the application logic.
   ============================================================================ */


/* ---------------------------------------------------------------------------
   Fence counter. Incremented on every job claim; a worker whose lease expired
   holds a lower token than its replacement and is rejected on write-back.
   A single-row counter rather than a sequence, which SQLite does not have.
   --------------------------------------------------------------------------- */
CREATE TABLE IF NOT EXISTS QA_FenceCounter
(
    Id          INTEGER NOT NULL PRIMARY KEY CHECK (Id = 1),
    NextValue   INTEGER NOT NULL
) STRICT;

INSERT OR IGNORE INTO QA_FenceCounter (Id, NextValue) VALUES (1, 1);


/* ===========================================================================
   QA_SamplingRuns — one row per reporting day per policy version.
   =========================================================================== */
CREATE TABLE IF NOT EXISTS QA_SamplingRuns
(
    RunId                   TEXT    NOT NULL PRIMARY KEY,
    ReportingDate           TEXT    NOT NULL,              -- 'YYYY-MM-DD', local date
    ReportingTimezone       TEXT    NOT NULL,              -- IANA name, e.g. 'Africa/Cairo'

    /* Actual UTC bounds of the reporting day, stored rather than recomputed so
       a later tzdata update cannot retroactively move a committed day's
       boundaries. DayHours is 23, 24 or 25 across an Egypt DST transition. */
    DayStartUtc             TEXT    NOT NULL,
    DayEndUtc               TEXT    NOT NULL,
    DayHours                REAL    GENERATED ALWAYS AS
                                ((julianday(DayEndUtc) - julianday(DayStartUtc)) * 24.0) STORED,

    SamplingPolicyVersion   TEXT    NOT NULL,
    BaseSeed                TEXT    NOT NULL,
    DailyTarget             INTEGER NOT NULL CHECK (DailyTarget > 0),
    SamplingIntervalMinutes INTEGER NOT NULL CHECK (SamplingIntervalMinutes > 0),
    SourceGraceMinutes      INTEGER NOT NULL CHECK (SourceGraceMinutes >= 0),
    ExcludeAdjacentChats    INTEGER NOT NULL CHECK (ExcludeAdjacentChats IN (0, 1)),
    MinChatGapMinutes       INTEGER NOT NULL CHECK (MinChatGapMinutes >= 0),

    RosterVersion           TEXT    NULL,
    RosterAgentCount        INTEGER NULL,

    Status                  TEXT    NOT NULL DEFAULT 'planning'
                                CHECK (Status IN ('planning','active','reconciling','closed','abandoned')),
    PlannedAtUtc            TEXT    NULL,
    ClosedAtUtc             TEXT    NULL,

    ReservedCount           INTEGER NOT NULL DEFAULT 0,
    CompletedCount          INTEGER NOT NULL DEFAULT 0,
    ShortfallCount          INTEGER NULL,
    ShortfallReason         TEXT    NULL,

    CreatedAtUtc            TEXT    NOT NULL,
    UpdatedAtUtc            TEXT    NOT NULL,

    CHECK (DayEndUtc > DayStartUtc)
) STRICT;

/* One live run per reporting day per policy version. A re-plan under a new
   policy version is allowed and tracked separately; a duplicate run under the
   same version is a bug, blocked here rather than in application code. */
CREATE UNIQUE INDEX IF NOT EXISTS UX_QA_Runs_Date_Policy
    ON QA_SamplingRuns (ReportingDate, SamplingPolicyVersion)
    WHERE Status <> 'abandoned';


/* ===========================================================================
   QA_AgentQuotas — per-agent budget for a run, with the reallocation audit.
   =========================================================================== */
CREATE TABLE IF NOT EXISTS QA_AgentQuotas
(
    RunId               TEXT    NOT NULL REFERENCES QA_SamplingRuns (RunId),
    AgentEmailAddress   TEXT    NOT NULL,              -- join key to DWH.AI.Agents
    AgentId             TEXT    NULL,
    AgentName           TEXT    NULL,

    /* The reporting day's own UTC bounds. Sampling covers every agent across
       the full day — there is no shift model (see 002_drop_shifts.sql). These
       are kept rather than derived so a 23- or 25-hour Cairo day is visible in
       the record instead of being inferred later. */
    WorkStartUtc        TEXT    NOT NULL,
    WorkEndUtc          TEXT    NOT NULL,
    WorkMinutes         INTEGER GENERATED ALWAYS AS
                            (CAST(ROUND((julianday(WorkEndUtc) - julianday(WorkStartUtc)) * 1440.0) AS INTEGER)) STORED,

    OriginalQuota       INTEGER NOT NULL CHECK (OriginalQuota >= 0),
    FinalQuota          INTEGER NOT NULL CHECK (FinalQuota >= 0),
    ReallocationReason  TEXT    NULL,

    /* Deterministic remainder ranking from the date-seeded agent order, kept
       so an audit can reproduce who received the +1. */
    RemainderRank       INTEGER NULL,

    ReservedCount       INTEGER NOT NULL DEFAULT 0,
    CompletedCount      INTEGER NOT NULL DEFAULT 0,

    CreatedAtUtc        TEXT    NOT NULL,

    /* WorkStartUtc is redundant in the key now that there is exactly one row
       per agent per run — it is retained only so that ledgers created before
       002 need no table rebuild. */
    PRIMARY KEY (RunId, AgentEmailAddress, WorkStartUtc),
    CHECK (WorkEndUtc > WorkStartUtc)
) STRICT;


/* ===========================================================================
   QA_SamplingIntervals — the dispatch unit. One row per agent per interval.
   =========================================================================== */
CREATE TABLE IF NOT EXISTS QA_SamplingIntervals
(
    IntervalRowId       INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    RunId               TEXT    NOT NULL REFERENCES QA_SamplingRuns (RunId),
    AgentEmailAddress   TEXT    NOT NULL,

    /* IntervalId comes from app.automation.calendar.Interval and feeds the
       sampling hash. Format '<startUtc>/<endUtc>' as yyyyMMddTHHmmZ. Changing
       the format changes every future selection, so it is pinned to the run's
       SamplingPolicyVersion. */
    IntervalId          TEXT    NOT NULL,
    IntervalIndex       INTEGER NOT NULL,
    IntervalStartUtc    TEXT    NOT NULL,
    IntervalEndUtc      TEXT    NOT NULL,
    IntervalMinutes     INTEGER GENERATED ALWAYS AS
                            (CAST(ROUND((julianday(IntervalEndUtc) - julianday(IntervalStartUtc)) * 1440.0) AS INTEGER)) STORED,

    /* IntervalEndUtc + grace. The dispatcher picks up any interval whose cutoff
       has passed and which is not yet terminal — which is what makes a missed
       scheduler run self-healing rather than a lost interval. */
    SnapshotCutoffUtc   TEXT    NOT NULL,

    Quota               INTEGER NOT NULL CHECK (Quota >= 0),
    SnapshotId          TEXT    NULL,
    SnapshotVersion     INTEGER NOT NULL DEFAULT 1,

    Status              TEXT    NOT NULL DEFAULT 'pending'
                            CHECK (Status IN ('pending','snapshotting','snapshotted','empty','failed')),
    CandidateCount      INTEGER NULL,
    SelectedCount       INTEGER NULL,
    ShortfallCount      INTEGER NULL,
    ShortfallReason     TEXT    NULL,

    SnapshottedAtUtc    TEXT    NULL,
    CreatedAtUtc        TEXT    NOT NULL,

    CHECK (IntervalEndUtc > IntervalStartUtc)
) STRICT;

/* An interval is snapshotted exactly once per run + agent. This is what stops
   a duplicate dispatcher pass from enqueuing the same interval twice. */
CREATE UNIQUE INDEX IF NOT EXISTS UX_QA_Int_Run_Agent_Interval
    ON QA_SamplingIntervals (RunId, AgentEmailAddress, IntervalId, SnapshotVersion);

/* The dispatcher's hot path: due and not terminal. */
CREATE INDEX IF NOT EXISTS IX_QA_Int_Due
    ON QA_SamplingIntervals (SnapshotCutoffUtc)
    WHERE Status IN ('pending','snapshotting');


/* ===========================================================================
   QA_SamplingCandidates — the frozen candidate snapshot.

   Every conversation considered for an interval, selected or not, with the
   rank hash and the sequence position the nonconsecutive rule reads. Keeping
   rejected candidates is what makes a selection auditable; without them
   "why was this chat not picked" is unanswerable.
   =========================================================================== */
CREATE TABLE IF NOT EXISTS QA_SamplingCandidates
(
    SnapshotId          TEXT    NOT NULL,
    ConversationId      TEXT    NOT NULL,
    RunId               TEXT    NOT NULL REFERENCES QA_SamplingRuns (RunId),
    AgentEmailAddress   TEXT    NOT NULL,
    IntervalId          TEXT    NOT NULL,

    /* Attribution (decision D2). AttributionRule records *how* the agent was
       chosen, so a later rule change is visible in the data rather than
       retroactively implied. */
    AttributionRule     TEXT    NOT NULL,
    HandlingTimeUtc     TEXT    NOT NULL,

    /* Chronological position among ALL attributable handled conversations for
       this agent and shift, including ineligible ones — "not back to back" is
       defined over the real sequence, not the eligible subset. Frozen at
       snapshot time; late arrivals never renumber it. */
    SequencePosition    INTEGER NOT NULL,

    RankHash            TEXT    NOT NULL CHECK (length(RankHash) = 64),

    IsEligible          INTEGER NOT NULL CHECK (IsEligible IN (0, 1)),
    EligibilityReason   TEXT    NULL,
    IsSelected          INTEGER NOT NULL DEFAULT 0 CHECK (IsSelected IN (0, 1)),
    SelectionReason     TEXT    NULL,

    ConversationState       TEXT NULL,
    SourceLastUpdatedUtc    TEXT NULL,
    IsLateArrival           INTEGER NOT NULL DEFAULT 0 CHECK (IsLateArrival IN (0, 1)),

    CreatedAtUtc        TEXT    NOT NULL,

    PRIMARY KEY (SnapshotId, ConversationId)
) STRICT;

/* Selection-order replay: ORDER BY RankHash, ConversationId. */
CREATE INDEX IF NOT EXISTS IX_QA_Cand_Rank
    ON QA_SamplingCandidates (SnapshotId, RankHash, ConversationId);

/* Spacing checks across intervals and midnight read by agent + sequence. */
CREATE INDEX IF NOT EXISTS IX_QA_Cand_Agent_Sequence
    ON QA_SamplingCandidates (RunId, AgentEmailAddress, SequencePosition);


/* ===========================================================================
   QA_EvaluationJobs — the work queue.
   =========================================================================== */
CREATE TABLE IF NOT EXISTS QA_EvaluationJobs
(
    JobId                   TEXT    NOT NULL PRIMARY KEY,
    RunId                   TEXT    NOT NULL REFERENCES QA_SamplingRuns (RunId),
    SnapshotId              TEXT    NOT NULL,
    ConversationId          TEXT    NOT NULL,
    AgentEmailAddress       TEXT    NOT NULL,
    IntervalId              TEXT    NOT NULL,

    /* Denormalized from the run so the uniqueness index below needs no join,
       and so a policy bump is visible on the job itself. */
    SamplingPolicyVersion   TEXT    NOT NULL,
    ReportingDate           TEXT    NOT NULL,

    Status                  TEXT    NOT NULL DEFAULT 'pending'
                                CHECK (Status IN ('pending','running','retry_wait','completed','failed','superseded')),
    Attempts                INTEGER NOT NULL DEFAULT 0,
    NextAttemptUtc          TEXT    NULL,

    /* Lease. FenceToken comes from QA_FenceCounter on each claim; a worker
       writing back with a token below the row's current one has been
       superseded and must not persist its result. */
    LeaseOwner              TEXT    NULL,
    LeaseToken              TEXT    NULL,
    LeaseExpiresUtc         TEXT    NULL,
    FenceToken              INTEGER NULL,

    /* Frozen evaluation input: the transcript is pinned on first successful
       retrieval so a retry cannot silently evaluate a conversation that
       changed in the meantime. */
    InputChecksum           TEXT    NULL,
    InputFrozenAtUtc        TEXT    NULL,
    SourceLastUpdatedUtc    TEXT    NULL,

    /* Provenance, for score comparability as the model and prompt change. */
    ModelName               TEXT    NULL,
    RubricVersion           TEXT    NULL,
    PromptVersion           TEXT    NULL,

    LastError               TEXT    NULL,
    FailureClass            TEXT    NULL CHECK (FailureClass IS NULL OR FailureClass IN ('transient','permanent','input')),

    StartedAtUtc            TEXT    NULL,
    CompletedAtUtc          TEXT    NULL,
    CreatedAtUtc            TEXT    NOT NULL
) STRICT;

/* ───────────────────────────────────────────────────────────────────────────
   The budget guarantee.

   One automatic evaluation per conversation per sampling-policy version,
   across all runs. This index — not application logic — is what makes
   "reservations never exceed the daily target in distinct selections" hold
   under reconciliation and under a second dispatcher pass. 'superseded' is
   excluded so an intentional reevaluation campaign can supersede a job and
   re-reserve the conversation under a new policy version.
   ─────────────────────────────────────────────────────────────────────────── */
CREATE UNIQUE INDEX IF NOT EXISTS UX_QA_Jobs_Conv_Policy
    ON QA_EvaluationJobs (ConversationId, SamplingPolicyVersion)
    WHERE Status <> 'superseded';

/* The worker's claim path. */
CREATE INDEX IF NOT EXISTS IX_QA_Jobs_Claimable
    ON QA_EvaluationJobs (Status, NextAttemptUtc)
    WHERE Status IN ('pending','retry_wait');

/* Lease-recovery sweep. */
CREATE INDEX IF NOT EXISTS IX_QA_Jobs_Expired_Leases
    ON QA_EvaluationJobs (LeaseExpiresUtc)
    WHERE Status = 'running';

/* Per-day coverage reporting. */
CREATE INDEX IF NOT EXISTS IX_QA_Jobs_Run_Agent
    ON QA_EvaluationJobs (RunId, AgentEmailAddress, Status);


/* ===========================================================================
   QA_ResultLinks — idempotent persistence WITHOUT a DWH schema change.

   The original design added a QA_Job_Id column to DWH.AI.Call_QA_Results and
   let a unique index reject a retry's duplicate insert. We cannot add that
   column, so the link lives here instead and the protocol does the work:

     1. Before inference, look for a completed link for this job.
        Present  -> the result is already in DWH. Mark the job completed and
                    DO NOT call the LLM again.

     2. Otherwise write an 'attempting' row, COMMITTED BEFORE the DWH insert,
        recording:
          PreInsertMaxVersion - MAX(Analysis_Version) for this Call_ID in DWH
                                right now
          AttemptStartedUtc   - the moment just before the insert
        This row is the crash-recovery evidence. Committing it first is the
        whole point: a crash after the DWH insert but before step 3 must still
        leave a trace that the insert may have happened.

     3. After insert_qa_result returns, update the row to 'linked' with the
        Analysis_Version and row count that landed.

   Recovery, when a retry finds an 'attempting' row:
        Re-read DWH for rows with this Call_ID where
            Analysis_Version > PreInsertMaxVersion
            AND Created_at    >= AttemptStartedUtc
        Found    -> the insert landed before the crash. Adopt that version,
                    mark 'linked', do not re-infer, do not re-insert.
        Not found-> the insert did not land. Safe to retry in full.

   RESIDUAL RISK, stated plainly: the Created_at window narrows but does not
   eliminate the case where a *manual* re-analysis of the same Call_ID lands
   inside the same window and is mistaken for ours. Call_QA_Results has no
   column we may add to disambiguate, and Created_at has one-second
   resolution. The window is the duration of one crashed attempt, and manual
   re-analysis of the exact conversation an automated job is evaluating at
   that exact moment is rare — but it is not impossible, and a run that hits
   it would under-count by one rather than duplicate. Revisit if DDL rights
   on DWH are ever granted.
   =========================================================================== */
CREATE TABLE IF NOT EXISTS QA_ResultLinks
(
    JobId               TEXT    NOT NULL PRIMARY KEY REFERENCES QA_EvaluationJobs (JobId),
    CallId              TEXT    NOT NULL,

    Status              TEXT    NOT NULL CHECK (Status IN ('attempting','linked','abandoned')),

    /* Crash-recovery evidence, written before the DWH insert. */
    PreInsertMaxVersion INTEGER NOT NULL,
    AttemptStartedUtc   TEXT    NOT NULL,

    /* Filled in after the insert succeeds. */
    AnalysisVersion     INTEGER NULL,
    RowsInserted        INTEGER NULL,
    LinkedAtUtc         TEXT    NULL,

    /* How the link was established: a normal insert, or recovery after a
       crash. Worth distinguishing — a rising 'recovered' count means the
       worker is dying mid-persist and that needs investigating. */
    LinkMethod          TEXT    NULL CHECK (LinkMethod IS NULL OR LinkMethod IN ('inserted','recovered')),

    Attempts            INTEGER NOT NULL DEFAULT 1,
    Notes               TEXT    NULL,

    CHECK (Status <> 'linked' OR (AnalysisVersion IS NOT NULL AND LinkedAtUtc IS NOT NULL))
) STRICT;

/* A Call_ID may legitimately appear more than once across policy versions, so
   this is an index, not a unique constraint. Used by recovery lookups. */
CREATE INDEX IF NOT EXISTS IX_QA_ResultLinks_Call
    ON QA_ResultLinks (CallId, Status);


/* ===========================================================================
   QA_RunCoverage — per-run coverage at a glance.

   Failures and shortfalls stay separate from QA scores (architecture §11): a
   quota shortfall is an operations problem, not a quality signal, and must
   never be read as one.
   =========================================================================== */
DROP VIEW IF EXISTS QA_RunCoverage;
CREATE VIEW QA_RunCoverage AS
SELECT
    r.RunId,
    r.ReportingDate,
    r.ReportingTimezone,
    r.DayHours,
    r.SamplingPolicyVersion,
    r.DailyTarget,
    r.Status                                                            AS RunStatus,
    COUNT(j.JobId)                                                      AS JobsReserved,
    SUM(CASE WHEN j.Status = 'completed'  THEN 1 ELSE 0 END)            AS JobsCompleted,
    SUM(CASE WHEN j.Status = 'failed'     THEN 1 ELSE 0 END)            AS JobsFailed,
    SUM(CASE WHEN j.Status = 'retry_wait' THEN 1 ELSE 0 END)            AS JobsAwaitingRetry,
    SUM(CASE WHEN j.Status IN ('pending','running') THEN 1 ELSE 0 END)  AS JobsOutstanding,
    r.DailyTarget - COUNT(j.JobId)                                      AS ReservationDeficit,
    COUNT(DISTINCT j.AgentEmailAddress)                                 AS AgentsCovered
FROM QA_SamplingRuns r
LEFT JOIN QA_EvaluationJobs j
       ON j.RunId = r.RunId
      AND j.Status <> 'superseded'
GROUP BY
    r.RunId, r.ReportingDate, r.ReportingTimezone, r.DayHours,
    r.SamplingPolicyVersion, r.DailyTarget, r.Status;


/* ===========================================================================
   Schema version marker.
   =========================================================================== */
CREATE TABLE IF NOT EXISTS QA_SchemaVersion
(
    Version     INTEGER NOT NULL PRIMARY KEY,
    AppliedAtUtc TEXT   NOT NULL,
    Description TEXT    NOT NULL
) STRICT;
