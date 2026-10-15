# NOT THE ACTIVE SCHEMA — reference only

These two T-SQL scripts require `CREATE TABLE` and `ALTER TABLE` rights on the
`DWH` database. **We do not have those rights**, so neither script is run and
neither should be handed to anyone to run as part of this work.

The live ledger is SQLite, in `../sqlite/`. See
`docs/AUTOMATION_IMPLEMENTATION_PLAN.md` §2 for what changes as a result —
in particular, `002_results_job_link.sql` adds a `QA_Job_Id` column to
`DWH.AI.Call_QA_Results`, which is exactly the thing we cannot do, so
idempotent persistence is solved differently (local link + reconciliation
against the version counter, rather than a unique index on the DWH table).

Keep these only in case DDL rights on `DWH` are granted later. The SQLite
schema is the one to change in the meantime; if these are ever revived, they
must be re-reconciled against it first.

`DWH.AI.Call_QA_Results` is read and inserted into by the existing
application. Automation must do the same and nothing more: **no DDL, no
ALTER, no schema change of any kind against DWH.**
