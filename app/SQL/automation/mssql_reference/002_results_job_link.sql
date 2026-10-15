/* ============================================================================
   002 — Link QA results to automation jobs
   Target: SQL Server 2016+   Table: DWH.AI.Call_QA_Results
   See docs/AUTOMATION_IMPLEMENTATION_PLAN.md §2.2 and §4

   What this solves
   ────────────────
   app/services/sql_helpers.py:236-247 resolves Analysis_Version as
   MAX(Analysis_Version) + 1 for the Call_ID. That is right for the manual
   dashboard — re-running an analysis should keep the old one — but it makes
   the automated worker non-idempotent: a job that crashes after the LLM call
   but before its status is committed will, on retry, insert a second version
   of the same evaluation and inflate the day's completed count.

   The fix is a nullable QA_Job_Id plus a filtered unique index. A retry's
   duplicate insert then fails at the database with error 2601, which the
   worker treats as "already persisted, succeed" rather than as an error. The
   guarantee lives in the index, not in application logic, so it also holds
   when worker concurrency is raised above 1.

   Backwards compatibility
   ───────────────────────
   The column is NULLABLE with no default. Every existing row keeps NULL, the
   filtered index ignores NULLs, and the manual path — which never sets it —
   is completely unaffected. No backfill, no downtime.

   Idempotent: safe to re-run.
   ============================================================================ */

SET NOCOUNT ON;
GO

IF OBJECT_ID('AI.Call_QA_Results', 'U') IS NULL
BEGIN
    RAISERROR('AI.Call_QA_Results does not exist in this database. Run this against the DWH database.', 16, 1);
    SET NOEXEC ON;
END
GO

/* ── The job link ─────────────────────────────────────────────────────────── */
IF COL_LENGTH('AI.Call_QA_Results', 'QA_Job_Id') IS NULL
    ALTER TABLE AI.Call_QA_Results ADD QA_Job_Id UNIQUEIDENTIFIER NULL;
GO

/* Model and rubric provenance, so automated scores stay comparable when the
   model or the prompt changes underneath them. Also nullable for existing rows. */
IF COL_LENGTH('AI.Call_QA_Results', 'Model_Name') IS NULL
    ALTER TABLE AI.Call_QA_Results ADD Model_Name NVARCHAR(128) NULL;
GO

IF COL_LENGTH('AI.Call_QA_Results', 'Rubric_Version') IS NULL
    ALTER TABLE AI.Call_QA_Results ADD Rubric_Version NVARCHAR(32) NULL;
GO

IF COL_LENGTH('AI.Call_QA_Results', 'Prompt_Version') IS NULL
    ALTER TABLE AI.Call_QA_Results ADD Prompt_Version NVARCHAR(32) NULL;
GO

/* Distinguishes automatic from manual rows without inspecting QA_Job_Id, for
   dashboards that must not mix the two populations. */
IF COL_LENGTH('AI.Call_QA_Results', 'Evaluation_Source') IS NULL
    ALTER TABLE AI.Call_QA_Results
        ADD Evaluation_Source NVARCHAR(16) NULL
            CONSTRAINT CK_Call_QA_Results_EvalSource
                CHECK (Evaluation_Source IS NULL OR Evaluation_Source IN ('manual','automatic','upload'));
GO


/* ───────────────────────────────────────────────────────────────────────────
   The idempotency guarantee.

   NOTE ON CARDINALITY: Call_QA_Results is denormalized — one row per
   compliance flag, with call-level fields repeated (see the module docstring
   in app/services/sql_helpers.py). A single job therefore legitimately writes
   SEVERAL rows. A unique index on QA_Job_Id alone would reject the second
   flag row of a perfectly normal multi-flag result.

   The uniqueness that actually matters is (QA_Job_Id, Analysis_Version): one
   job produces exactly one version, however many flag rows that version
   contains. A retry re-inserting the same version is rejected; a multi-flag
   result inserts freely.
   ─────────────────────────────────────────────────────────────────────────── */
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UX_Call_QA_Results_Job_Version'
                 AND object_id = OBJECT_ID('AI.Call_QA_Results'))
    CREATE UNIQUE INDEX UX_Call_QA_Results_Job_Version
        ON AI.Call_QA_Results (QA_Job_Id, Analysis_Version)
        WHERE QA_Job_Id IS NOT NULL;
GO

/* Lookup from a job to its persisted rows, used by the worker to reconcile a
   result that may already have been written before a crash. */
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Call_QA_Results_Job'
                 AND object_id = OBJECT_ID('AI.Call_QA_Results'))
    CREATE INDEX IX_Call_QA_Results_Job
        ON AI.Call_QA_Results (QA_Job_Id)
        INCLUDE (Call_ID, Analysis_Version, Overall_Assessment)
        WHERE QA_Job_Id IS NOT NULL;
GO

SET NOEXEC OFF;
GO
