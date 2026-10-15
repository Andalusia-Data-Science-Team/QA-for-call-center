/* ============================================================================
   001 — Automated QA sampling ledger
   Target: SQL Server 2016+   Schema: DWH.AI
   See docs/AUTOMATION_IMPLEMENTATION_PLAN.md §2.1

   Idempotent: safe to re-run. Creates nothing that the existing manual
   dashboard path reads or writes — DWH.AI.Call_QA_Results is untouched here
   (see 002_results_job_link.sql).

   Timezone contract: every datetime column in this file is UTC, stored as
   DATETIME2(3). The reporting-day *label* lives in QA_SamplingRuns.ReportingDate
   as a DATE together with the IANA zone name that produced it. Nothing in this
   schema applies an offset; app/automation/calendar.py owns that conversion,
   because Africa/Cairo is UTC+2 or UTC+3 depending on the date and a stored
   fixed offset would be wrong for half the year.
   ============================================================================ */

SET NOCOUNT ON;
GO

IF SCHEMA_ID('AI') IS NULL
    EXEC('CREATE SCHEMA AI');
GO

/* ---------------------------------------------------------------------------
   Fencing sequence. Handed to a worker when it claims a job; a worker whose
   lease expired holds a lower token than its replacement and is rejected on
   write-back. Monotonic across the whole ledger, so it also orders claims.
   --------------------------------------------------------------------------- */
IF NOT EXISTS (SELECT 1 FROM sys.sequences WHERE name = 'QA_FenceSeq' AND schema_id = SCHEMA_ID('AI'))
    CREATE SEQUENCE AI.QA_FenceSeq AS BIGINT START WITH 1 INCREMENT BY 1 NO CYCLE CACHE 50;
GO


/* ===========================================================================
   QA_SamplingRuns — one row per reporting day per policy version.
   =========================================================================== */
IF OBJECT_ID('AI.QA_SamplingRuns', 'U') IS NULL
BEGIN
    CREATE TABLE AI.QA_SamplingRuns
    (
        RunId                   UNIQUEIDENTIFIER NOT NULL
                                    CONSTRAINT DF_QA_Runs_RunId DEFAULT NEWSEQUENTIALID(),
        ReportingDate           DATE            NOT NULL,
        ReportingTimezone       NVARCHAR(64)    NOT NULL,   -- IANA name, e.g. 'Africa/Cairo'

        /* Actual UTC bounds of the reporting day. Stored, not recomputed, so a
           later tzdata update cannot retroactively move a committed day's
           boundaries. DayHours is 23, 24 or 25 across a DST transition. */
        DayStartUtc             DATETIME2(3)    NOT NULL,
        DayEndUtc               DATETIME2(3)    NOT NULL,
        DayHours                AS (CAST(DATEDIFF(MINUTE, DayStartUtc, DayEndUtc) AS DECIMAL(6,2)) / 60.0) PERSISTED,

        SamplingPolicyVersion   NVARCHAR(32)    NOT NULL,
        BaseSeed                NVARCHAR(128)   NOT NULL,
        DailyTarget             INT             NOT NULL,
        SamplingIntervalMinutes INT             NOT NULL,
        SourceGraceMinutes      INT             NOT NULL,
        ExcludeAdjacentChats    BIT             NOT NULL,
        MinChatGapMinutes       INT             NOT NULL,

        RosterVersion           NVARCHAR(64)    NULL,
        RosterAgentCount        INT             NULL,

        Status                  NVARCHAR(20)    NOT NULL
                                    CONSTRAINT DF_QA_Runs_Status DEFAULT 'planning',
        PlannedAtUtc            DATETIME2(3)    NULL,
        ClosedAtUtc             DATETIME2(3)    NULL,

        /* Final accounting, written by the reconciler. */
        ReservedCount           INT             NOT NULL CONSTRAINT DF_QA_Runs_Reserved DEFAULT 0,
        CompletedCount          INT             NOT NULL CONSTRAINT DF_QA_Runs_Completed DEFAULT 0,
        ShortfallCount          INT             NULL,
        ShortfallReason         NVARCHAR(1000)  NULL,

        CreatedAtUtc            DATETIME2(3)    NOT NULL CONSTRAINT DF_QA_Runs_Created DEFAULT SYSUTCDATETIME(),
        UpdatedAtUtc            DATETIME2(3)    NOT NULL CONSTRAINT DF_QA_Runs_Updated DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_QA_SamplingRuns PRIMARY KEY CLUSTERED (RunId),
        CONSTRAINT CK_QA_Runs_Status CHECK (Status IN ('planning','active','reconciling','closed','abandoned')),
        CONSTRAINT CK_QA_Runs_Bounds CHECK (DayEndUtc > DayStartUtc),
        CONSTRAINT CK_QA_Runs_Target CHECK (DailyTarget > 0)
    );

    /* One active run per reporting day per policy version. A re-plan under a
       new policy version is allowed and tracked separately; a duplicate run
       under the same version is a bug and is blocked here rather than in code. */
    CREATE UNIQUE INDEX UX_QA_Runs_Date_Policy
        ON AI.QA_SamplingRuns (ReportingDate, SamplingPolicyVersion)
        WHERE Status <> 'abandoned';
END
GO


/* ===========================================================================
   QA_AgentQuotas — per-agent budget for a run, with the reallocation audit.
   =========================================================================== */
IF OBJECT_ID('AI.QA_AgentQuotas', 'U') IS NULL
BEGIN
    CREATE TABLE AI.QA_AgentQuotas
    (
        RunId               UNIQUEIDENTIFIER NOT NULL,
        AgentEmailAddress   NVARCHAR(200)   NOT NULL,   -- join key to DWH.AI.Agents
        AgentId             NVARCHAR(64)    NULL,
        AgentName           NVARCHAR(255)   NULL,

        /* Working period for this reporting day, already clipped to the day's
           UTC bounds. An overnight shift produces one row per day it touches. */
        WorkStartUtc        DATETIME2(3)    NOT NULL,
        WorkEndUtc          DATETIME2(3)    NOT NULL,
        WorkMinutes         AS (DATEDIFF(MINUTE, WorkStartUtc, WorkEndUtc)) PERSISTED,
        WorkPeriodSource    NVARCHAR(32)    NOT NULL,    -- 'roster' | 'config' | 'observed'

        OriginalQuota       INT             NOT NULL,
        FinalQuota          INT             NOT NULL,
        ReallocationReason  NVARCHAR(500)   NULL,

        /* Deterministic remainder ranking from the date-seeded agent order,
           retained so an audit can reproduce who got the +1. */
        RemainderRank       INT             NULL,

        ReservedCount       INT             NOT NULL CONSTRAINT DF_QA_Quota_Reserved DEFAULT 0,
        CompletedCount      INT             NOT NULL CONSTRAINT DF_QA_Quota_Completed DEFAULT 0,

        CreatedAtUtc        DATETIME2(3)    NOT NULL CONSTRAINT DF_QA_Quota_Created DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_QA_AgentQuotas PRIMARY KEY CLUSTERED (RunId, AgentEmailAddress, WorkStartUtc),
        CONSTRAINT FK_QA_Quota_Run FOREIGN KEY (RunId) REFERENCES AI.QA_SamplingRuns (RunId),
        CONSTRAINT CK_QA_Quota_Bounds CHECK (WorkEndUtc > WorkStartUtc),
        CONSTRAINT CK_QA_Quota_NonNegative CHECK (OriginalQuota >= 0 AND FinalQuota >= 0),
        CONSTRAINT CK_QA_Quota_Source CHECK (WorkPeriodSource IN ('roster','config','observed'))
    );
END
GO


/* ===========================================================================
   QA_SamplingIntervals — the dispatch unit. One row per agent per interval.
   =========================================================================== */
IF OBJECT_ID('AI.QA_SamplingIntervals', 'U') IS NULL
BEGIN
    CREATE TABLE AI.QA_SamplingIntervals
    (
        IntervalRowId       BIGINT IDENTITY(1,1) NOT NULL,
        RunId               UNIQUEIDENTIFIER NOT NULL,
        AgentEmailAddress   NVARCHAR(200)   NOT NULL,

        /* IntervalId is produced by app.automation.calendar.Interval and feeds
           the sampling hash. Format '<startUtc>/<endUtc>' as
           yyyyMMddTHHmmZ. Changing the format changes every future
           selection, so it is pinned to SamplingPolicyVersion on the run. */
        IntervalId          NVARCHAR(64)    NOT NULL,
        IntervalIndex       INT             NOT NULL,
        IntervalStartUtc    DATETIME2(3)    NOT NULL,
        IntervalEndUtc      DATETIME2(3)    NOT NULL,
        IntervalMinutes     AS (DATEDIFF(MINUTE, IntervalStartUtc, IntervalEndUtc)) PERSISTED,

        /* IntervalEndUtc + grace. The dispatcher picks up any interval whose
           cutoff has passed and which is not yet terminal — which is what makes
           a missed scheduler run self-healing rather than a lost interval. */
        SnapshotCutoffUtc   DATETIME2(3)    NOT NULL,

        Quota               INT             NOT NULL,
        SnapshotId          UNIQUEIDENTIFIER NULL,
        SnapshotVersion     INT             NOT NULL CONSTRAINT DF_QA_Int_SnapVer DEFAULT 1,

        Status              NVARCHAR(20)    NOT NULL CONSTRAINT DF_QA_Int_Status DEFAULT 'pending',
        CandidateCount      INT             NULL,
        SelectedCount       INT             NULL,
        ShortfallCount      INT             NULL,
        ShortfallReason     NVARCHAR(500)   NULL,

        SnapshottedAtUtc    DATETIME2(3)    NULL,
        CreatedAtUtc        DATETIME2(3)    NOT NULL CONSTRAINT DF_QA_Int_Created DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_QA_SamplingIntervals PRIMARY KEY CLUSTERED (IntervalRowId),
        CONSTRAINT FK_QA_Int_Run FOREIGN KEY (RunId) REFERENCES AI.QA_SamplingRuns (RunId),
        CONSTRAINT CK_QA_Int_Status CHECK (Status IN ('pending','snapshotting','snapshotted','empty','failed')),
        CONSTRAINT CK_QA_Int_Bounds CHECK (IntervalEndUtc > IntervalStartUtc),
        CONSTRAINT CK_QA_Int_Quota CHECK (Quota >= 0)
    );

    /* An interval is snapshotted exactly once per run+agent. This is what stops
       a duplicate dispatcher run from enqueuing the same interval twice. */
    CREATE UNIQUE INDEX UX_QA_Int_Run_Agent_Interval
        ON AI.QA_SamplingIntervals (RunId, AgentEmailAddress, IntervalId, SnapshotVersion);

    /* The dispatcher's hot path: due and not terminal. */
    CREATE INDEX IX_QA_Int_Due
        ON AI.QA_SamplingIntervals (SnapshotCutoffUtc)
        INCLUDE (RunId, AgentEmailAddress, Quota)
        WHERE Status IN ('pending','snapshotting');
END
GO


/* ===========================================================================
   QA_SamplingCandidates — the frozen candidate snapshot.

   Every conversation considered for an interval, selected or not, with the
   rank hash and the sequence position that the nonconsecutive rule reads.
   Retaining rejected candidates is what makes a selection auditable and
   reproducible; without them "why was this chat not picked" is unanswerable.
   =========================================================================== */
IF OBJECT_ID('AI.QA_SamplingCandidates', 'U') IS NULL
BEGIN
    CREATE TABLE AI.QA_SamplingCandidates
    (
        SnapshotId          UNIQUEIDENTIFIER NOT NULL,
        ConversationId      UNIQUEIDENTIFIER NOT NULL,
        RunId               UNIQUEIDENTIFIER NOT NULL,
        AgentEmailAddress   NVARCHAR(200)   NOT NULL,
        IntervalId          NVARCHAR(64)    NOT NULL,

        /* Attribution (decision D2). AttributionRule records *how* the agent was
           chosen, so a later rule change is visible in the data rather than
           retroactively implied. */
        AttributionRule     NVARCHAR(64)    NOT NULL,
        HandlingTimeUtc     DATETIME2(3)    NOT NULL,

        /* Chronological position among ALL attributable handled conversations
           for this agent and shift, including ineligible ones — the "not back
           to back" rule is defined over the real sequence, not the eligible
           subset. Frozen at snapshot; late arrivals never renumber it. */
        SequencePosition    INT             NOT NULL,

        RankHash            CHAR(64)        NOT NULL,   -- SHA-256 hex

        IsEligible          BIT             NOT NULL,
        EligibilityReason   NVARCHAR(200)   NULL,       -- why excluded, when not eligible
        IsSelected          BIT             NOT NULL CONSTRAINT DF_QA_Cand_Selected DEFAULT 0,
        SelectionReason     NVARCHAR(200)   NULL,       -- e.g. 'adjacent_to_selected', 'quota_met'

        ConversationState   NVARCHAR(64)    NULL,
        SourceLastUpdatedUtc DATETIME2(3)   NULL,
        IsLateArrival       BIT             NOT NULL CONSTRAINT DF_QA_Cand_Late DEFAULT 0,

        CreatedAtUtc        DATETIME2(3)    NOT NULL CONSTRAINT DF_QA_Cand_Created DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_QA_SamplingCandidates PRIMARY KEY CLUSTERED (SnapshotId, ConversationId),
        CONSTRAINT FK_QA_Cand_Run FOREIGN KEY (RunId) REFERENCES AI.QA_SamplingRuns (RunId)
    );

    /* Selection order replay: ORDER BY RankHash, ConversationId. */
    CREATE INDEX IX_QA_Cand_Rank
        ON AI.QA_SamplingCandidates (SnapshotId, RankHash, ConversationId);

    /* Spacing checks across intervals and midnight read by agent + sequence. */
    CREATE INDEX IX_QA_Cand_Agent_Sequence
        ON AI.QA_SamplingCandidates (RunId, AgentEmailAddress, SequencePosition)
        INCLUDE (ConversationId, IsSelected);
END
GO


/* ===========================================================================
   QA_EvaluationJobs — the work queue.
   =========================================================================== */
IF OBJECT_ID('AI.QA_EvaluationJobs', 'U') IS NULL
BEGIN
    CREATE TABLE AI.QA_EvaluationJobs
    (
        JobId                   UNIQUEIDENTIFIER NOT NULL
                                    CONSTRAINT DF_QA_Jobs_JobId DEFAULT NEWID(),
        RunId                   UNIQUEIDENTIFIER NOT NULL,
        SnapshotId              UNIQUEIDENTIFIER NOT NULL,
        ConversationId          UNIQUEIDENTIFIER NOT NULL,
        AgentEmailAddress       NVARCHAR(200)   NOT NULL,
        IntervalId              NVARCHAR(64)    NOT NULL,

        /* Denormalized from the run so the uniqueness index below does not need
           a join, and so a policy bump is visible on the job itself. */
        SamplingPolicyVersion   NVARCHAR(32)    NOT NULL,
        ReportingDate           DATE            NOT NULL,

        Status                  NVARCHAR(20)    NOT NULL CONSTRAINT DF_QA_Jobs_Status DEFAULT 'pending',
        Attempts                INT             NOT NULL CONSTRAINT DF_QA_Jobs_Attempts DEFAULT 0,
        NextAttemptUtc          DATETIME2(3)    NULL,

        /* Lease. FenceToken comes from AI.QA_FenceSeq on each claim; a worker
           writing back with a token lower than the row's current one has been
           superseded and must not persist its result. */
        LeaseOwner              NVARCHAR(128)   NULL,
        LeaseToken              UNIQUEIDENTIFIER NULL,
        LeaseExpiresUtc         DATETIME2(3)    NULL,
        FenceToken              BIGINT          NULL,

        /* Frozen evaluation input (architecture §9): the transcript is pinned on
           first successful retrieval so a retry cannot silently evaluate a
           different conversation that changed in the meantime. */
        InputChecksum           CHAR(64)        NULL,
        InputFrozenAtUtc        DATETIME2(3)    NULL,
        SourceLastUpdatedUtc    DATETIME2(3)    NULL,

        /* Provenance of the result, for score comparability over time. */
        ModelName               NVARCHAR(128)   NULL,
        RubricVersion           NVARCHAR(32)    NULL,
        PromptVersion           NVARCHAR(32)    NULL,

        LastError               NVARCHAR(2000)  NULL,
        FailureClass            NVARCHAR(32)    NULL,   -- 'transient' | 'permanent' | 'input'

        StartedAtUtc            DATETIME2(3)    NULL,
        CompletedAtUtc          DATETIME2(3)    NULL,
        CreatedAtUtc            DATETIME2(3)    NOT NULL CONSTRAINT DF_QA_Jobs_Created DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_QA_EvaluationJobs PRIMARY KEY CLUSTERED (JobId),
        CONSTRAINT FK_QA_Jobs_Run FOREIGN KEY (RunId) REFERENCES AI.QA_SamplingRuns (RunId),
        CONSTRAINT CK_QA_Jobs_Status CHECK (Status IN ('pending','running','retry_wait','completed','failed','superseded')),
        CONSTRAINT CK_QA_Jobs_FailureClass CHECK (FailureClass IS NULL OR FailureClass IN ('transient','permanent','input'))
    );

    /* ───────────────────────────────────────────────────────────────────────
       The budget guarantee.

       One automatic evaluation per conversation per sampling-policy version,
       across all runs. This index — not application logic — is what makes
       "reservations never exceed 300 distinct selections" hold under
       concurrent dispatchers and under reconciliation. 'superseded' is
       excluded so an intentional reevaluation campaign can supersede a job
       and re-reserve the conversation under a new policy version.
       ─────────────────────────────────────────────────────────────────────── */
    CREATE UNIQUE INDEX UX_QA_Jobs_Conv_Policy
        ON AI.QA_EvaluationJobs (ConversationId, SamplingPolicyVersion)
        WHERE Status <> 'superseded';

    /* The worker's claim path. READPAST + this index keeps claiming O(1)
       when concurrency is raised above 1. */
    CREATE INDEX IX_QA_Jobs_Claimable
        ON AI.QA_EvaluationJobs (Status, NextAttemptUtc)
        INCLUDE (RunId, ConversationId, AgentEmailAddress)
        WHERE Status IN ('pending','retry_wait');

    /* Lease recovery sweep. */
    CREATE INDEX IX_QA_Jobs_Expired_Leases
        ON AI.QA_EvaluationJobs (LeaseExpiresUtc)
        WHERE Status = 'running';

    /* Per-day coverage reporting. */
    CREATE INDEX IX_QA_Jobs_Run_Agent
        ON AI.QA_EvaluationJobs (RunId, AgentEmailAddress, Status);
END
GO


/* ===========================================================================
   Operational view: per-run coverage at a glance. Failures and shortfalls are
   kept separate from QA scores, per architecture §11 — a quota shortfall is an
   operations problem, not a quality signal, and must never be read as one.
   =========================================================================== */
CREATE OR ALTER VIEW AI.QA_RunCoverage
AS
SELECT
    r.RunId,
    r.ReportingDate,
    r.ReportingTimezone,
    r.DayHours,
    r.SamplingPolicyVersion,
    r.DailyTarget,
    r.Status                                                        AS RunStatus,
    COUNT(j.JobId)                                                  AS JobsReserved,
    SUM(CASE WHEN j.Status = 'completed'  THEN 1 ELSE 0 END)        AS JobsCompleted,
    SUM(CASE WHEN j.Status = 'failed'     THEN 1 ELSE 0 END)        AS JobsFailed,
    SUM(CASE WHEN j.Status = 'retry_wait' THEN 1 ELSE 0 END)        AS JobsAwaitingRetry,
    SUM(CASE WHEN j.Status IN ('pending','running') THEN 1 ELSE 0 END) AS JobsOutstanding,
    r.DailyTarget - COUNT(j.JobId)                                  AS ReservationDeficit,
    COUNT(DISTINCT j.AgentEmailAddress)                             AS AgentsCovered
FROM AI.QA_SamplingRuns r
LEFT JOIN AI.QA_EvaluationJobs j
       ON j.RunId = r.RunId
      AND j.Status <> 'superseded'
GROUP BY
    r.RunId, r.ReportingDate, r.ReportingTimezone, r.DayHours,
    r.SamplingPolicyVersion, r.DailyTarget, r.Status;
GO
