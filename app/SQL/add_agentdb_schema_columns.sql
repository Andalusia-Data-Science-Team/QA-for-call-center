-- Migration for the expanded QA results table schema.
-- Run this once against the target database before using the updated insert flow.

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'Conversation_Link') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD Conversation_Link NVARCHAR(MAX) NULL;

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'Agent_Classification') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD Agent_Classification NVARCHAR(10) NULL;

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'Profiling_Comment') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD Profiling_Comment NVARCHAR(100) NULL;

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'BU') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD BU NVARCHAR(20) NULL;

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'Agent_Email_Address') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD Agent_Email_Address NVARCHAR(255) NULL;

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'Supervisor_Name') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD Supervisor_Name NVARCHAR(255) NULL;

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'Supervisor_Email_Address') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD Supervisor_Email_Address NVARCHAR(255) NULL;

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'Coaching_Status') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD Coaching_Status NVARCHAR(50) NULL;

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'Escalated') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD Escalated BIT NOT NULL CONSTRAINT DF_Call_QA_Results_Escalated DEFAULT (0);

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'QA_Reviewed') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD QA_Reviewed BIT NOT NULL CONSTRAINT DF_Call_QA_Results_QA_Reviewed DEFAULT (0);

IF COL_LENGTH('[DWH].[AI].[Call_QA_Results]', 'QA_Review_Comment') IS NULL
    ALTER TABLE [DWH].[AI].[Call_QA_Results]
    ADD QA_Review_Comment NVARCHAR(MAX) NULL;
