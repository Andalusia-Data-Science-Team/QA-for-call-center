-- Run locally before deploying code that writes the two new flag labels.
-- Additive migration only: existing rows and assessment text are untouched.
USE [DWH];
GO

IF COL_LENGTH(N'AI.Call_QA_Results', N'Attribute') IS NULL
    ALTER TABLE [AI].[Call_QA_Results] ADD [Attribute] NVARCHAR(500) NULL;
GO

IF COL_LENGTH(N'AI.Call_QA_Results', N'Comment') IS NULL
    ALTER TABLE [AI].[Call_QA_Results] ADD [Comment] NVARCHAR(2000) NULL;
GO
