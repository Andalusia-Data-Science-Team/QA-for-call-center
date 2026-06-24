-- ============================================================================
-- QA System Database Schema — Andalusia Hospitals Call Center
-- Target: SQL Server 2016+ (uses native JSON via ISJSON/JSON_VALUE,
-- computed PERSISTED columns, filtered indexes)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 0. agents — lookup table (recommended to avoid free-text agent_name drift
--    across DotCare / WhatsApp / Agora / Avaya sources)
-- ----------------------------------------------------------------------------
CREATE TABLE agents (
    agent_id        BIGINT IDENTITY(1,1) PRIMARY KEY,
    agent_name      NVARCHAR(255)  NOT NULL UNIQUE,
    department      NVARCHAR(100)  NULL,
    is_active       BIT            NOT NULL DEFAULT 1,
    created_at      DATETIME2(3)   NOT NULL DEFAULT SYSUTCDATETIME()
);

-- ============================================================================
-- 1. call_qa_results — ONE ROW PER ANALYZED CALL (immutable audit record)
--    Never UPDATE business fields after insert; if a call is re-analyzed,
--    insert a new row (add a `superseded_by` column later if needed).
-- ============================================================================
CREATE TABLE call_qa_results (
    id                          BIGINT IDENTITY(1,1) PRIMARY KEY,
    call_id                     NVARCHAR(128)  NOT NULL UNIQUE,

    -- Metadata sourced from CallTranscript INPUT, not from LLM output
    agent_id                    BIGINT         NULL REFERENCES agents(agent_id),
    agent_name                  NVARCHAR(255)  NOT NULL,  -- denormalized snapshot for history
    department                  NVARCHAR(100)  NULL,
    channel                     NVARCHAR(20)   NULL
                                 CHECK (channel IN ('dotcare', 'whatsapp', 'agora', 'voice', 'unknown')),
    call_date                   DATETIME2(3)   NOT NULL,
    call_duration_seconds       INT            NULL,

    -- QAAnalysisResult fields (LLM + deterministic pipeline output)
    overall_assessment          NVARCHAR(20)   NOT NULL
                                 CHECK (overall_assessment IN ('pass', 'needs_review', 'escalate', 'error')),
    assessment_reasoning        NVARCHAR(MAX)  NOT NULL,
    professionalism_score       DECIMAL(4,3)   NULL CHECK (professionalism_score BETWEEN 0 AND 1),
    accuracy_score               DECIMAL(4,3)  NULL CHECK (accuracy_score BETWEEN 0 AND 1),
    resolution_score              DECIMAL(4,3) NULL CHECK (resolution_score BETWEEN 0 AND 1),
    strengths                      NVARCHAR(MAX) NULL CHECK (strengths IS NULL OR ISJSON(strengths) = 1),
    improvements                    NVARCHAR(MAX) NULL CHECK (improvements IS NULL OR ISJSON(improvements) = 1),
    escalation_required               BIT NOT NULL DEFAULT 0,
    escalation_reason                   NVARCHAR(MAX) NULL,

    -- Deterministic weighted score from performance_scoring node (0-100 scale)
    weighted_score                        DECIMAL(5,2) NULL,
    passed_threshold                        AS (CASE WHEN weighted_score >= 85.0 THEN 1 ELSE 0 END) PERSISTED,
    
);

CREATE INDEX idx_call_qa_agent_date  ON call_qa_results (agent_name, call_date);
CREATE INDEX idx_call_qa_date        ON call_qa_results (call_date);
CREATE INDEX idx_call_qa_assessment  ON call_qa_results (overall_assessment);
CREATE INDEX idx_call_qa_escalation  ON call_qa_results (escalation_required) WHERE escalation_required = 1;

-- ============================================================================
-- 2. compliance_flags — normalized 1:N child of call_qa_results
-- ============================================================================
CREATE TABLE compliance_flags (
    id                  BIGINT IDENTITY(1,1) PRIMARY KEY,
    call_result_id      BIGINT        NOT NULL REFERENCES call_qa_results(id) ON DELETE CASCADE,
    type                 NVARCHAR(10) NOT NULL CHECK (type IN ('C2Com', 'C2C', 'C2B', 'NC')),
    severity             NVARCHAR(10) NOT NULL CHECK (severity IN ('critical', 'moderate', 'minor', 'positive')),
    violation_id          NVARCHAR(50) NULL,  -- e.g. 'C2Com_001' — see note below
    description             NVARCHAR(MAX) NOT NULL,
    transcript_excerpt        NVARCHAR(MAX) NULL,
    created_at                  DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
);

-- NOTE: your current prompt schema doesn't ask the LLM for a specific
-- violation_id, only `type` + free-text `description`. Add a violation_id
-- field to the OUTPUT SCHEMA in qa_prompt.py if you want the LLM to return
-- e.g. "C2Com_001" directly, rather than parsing it back out of description.

CREATE INDEX idx_flags_call_result    ON compliance_flags (call_result_id);
CREATE INDEX idx_flags_type_severity  ON compliance_flags (type, severity);
CREATE INDEX idx_flags_violation_id   ON compliance_flags (violation_id);

-- ============================================================================
-- 3. agent_monthly_scores — SEPARATE rollup table
--    Populated by a scheduled job (nightly / end-of-month) that aggregates
--    call_qa_results for the period and applies your monthly scoring formula.
-- ============================================================================
CREATE TABLE agent_monthly_scores (
    id                          BIGINT IDENTITY(1,1) PRIMARY KEY,
    agent_id                    BIGINT        NULL REFERENCES agents(agent_id),
    agent_name                  NVARCHAR(255) NOT NULL,
    year_month                  DATE          NOT NULL,  -- always 1st-of-month, e.g. 2026-06-01

    total_calls                 INT NOT NULL,
    calls_pass                  INT NOT NULL DEFAULT 0,
    calls_needs_review          INT NOT NULL DEFAULT 0,
    calls_escalate              INT NOT NULL DEFAULT 0,
    calls_error                 INT NOT NULL DEFAULT 0,

    avg_professionalism_score   DECIMAL(4,3) NULL,
    avg_accuracy_score           DECIMAL(4,3) NULL,
    avg_resolution_score          DECIMAL(4,3) NULL,

    critical_flags_count           INT NOT NULL DEFAULT 0,
    moderate_flags_count            INT NOT NULL DEFAULT 0,
    minor_flags_count                INT NOT NULL DEFAULT 0,
    positive_flags_count              INT NOT NULL DEFAULT 0,
    c2com_violations_count             INT NOT NULL DEFAULT 0,  -- privacy/PHI pillar, tracked separately

    monthly_overall_score                DECIMAL(5,2) NOT NULL,
    passed_monthly_threshold               AS (CASE WHEN monthly_overall_score >= 85.0 THEN 1 ELSE 0 END) PERSISTED,

    formula_version                          NVARCHAR(20) NOT NULL,  -- ties to compliance_regulations.yaml version
    computed_at                                 DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),

    CONSTRAINT uq_agent_month UNIQUE (agent_name, year_month)
);

CREATE INDEX idx_monthly_agent_month ON agent_monthly_scores (agent_name, year_month);

-- ============================================================================
-- Example: the kind of query agent_monthly_scores exists to make cheap
-- ============================================================================
-- SELECT agent_name, year_month, monthly_overall_score, passed_monthly_threshold
-- FROM agent_monthly_scores
-- WHERE year_month = '2026-06-01'
-- ORDER BY monthly_overall_score DESC;