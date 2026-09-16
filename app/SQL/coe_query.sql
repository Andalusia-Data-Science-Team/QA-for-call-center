-- Authoritative CRM reference data for the four supported Centers of
-- Excellence (COE). Used by app/service_hub/crm_coe.py to feed COE
-- validation (app/service_hub/coe_validation.py + app.agent.nodes.
-- infer_coe_validation) with COE names, business unit, specialties,
-- clinic members, clinic leader/coordinator, and the approved Arabic
-- scripts. NOTE: cr301_coemembers/cr301_clinicalleader are NOT used to
-- infer the approved *primary* doctor for a new COE booking — that list
-- is authoritative/hardcoded in coe_validation.py per business rules —
-- this query only supplies COE-level reference/context data.
SELECT
    cr301_coeclinicname       AS Clinic_Name,
    cr301_clinicalleader      AS Clinic_Leader,
    cr301_txtsubspecialty     AS Specialty,
    cr301_businessunitname    AS BU,
    cr301_clinicalcoordinator AS Clinic_Coordinator,
    cr301_coemembers          AS Member,
    cr301_arabicscript        AS Script_AR
FROM cr301_coelist
WHERE cr301_businessunitname = 'AHJ'
  AND cr301_coeclinicname IN ('IBD', 'Headache', 'Diabetes', 'Asthma');
