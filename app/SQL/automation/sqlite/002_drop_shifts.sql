/* ============================================================================
   002 — Remove the shift/working-period concept
   Local SQLite ledger only. Nothing in DWH is touched.

   WHY
   ───
   The shift model existed to allocate each agent's quota across the hours
   they actually worked. There is no shift feed (DWH.AI.Agents has no
   working-period columns and we cannot add any), so in practice every agent
   was given the whole reporting day anyway via a default. Carrying an
   abstraction that only ever produced one value — with a config file, three
   implementations and a provenance column to record which guess was used —
   was complexity with no payoff.

   Sampling now covers each agent across the full reporting day, which is
   what the architecture asks for: "covering the full 24-hour operating day".

   WorkStartUtc and WorkEndUtc are KEPT. They are now always the reporting
   day's own bounds, which still matters for audit: a 23- or 25-hour Cairo day
   must be visible in the record rather than inferred.

   Idempotent: safe to re-run.
   ============================================================================ */

/* SQLite 3.35+ supports DROP COLUMN. WorkPeriodSource is not part of the
   primary key or any index, so the drop is clean. Guarded by a pragma check
   in app/automation/db.py, which skips this file when the column is already
   gone — SQLite has no 'DROP COLUMN IF EXISTS'. */
ALTER TABLE QA_AgentQuotas DROP COLUMN WorkPeriodSource;
