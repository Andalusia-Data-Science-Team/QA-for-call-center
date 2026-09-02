-- Authoritative CRM doctor reference data for deterministic + semantic QA
-- validation. Deliberately NOT filtered by cr301_opdflag here — a doctor
-- who isn't flagged OPD (e.g. home-care/other non-OPD service contexts)
-- can still be a genuine, resolvable doctor; cr301_opdflag is fetched and
-- carried as informational metadata only (see doctor_validation.py),
-- never as a searchability gate. Active status and the supported-BU
-- allowlist are still interpreted in Python (doctor_validation.py)
-- because they depend on this project's own BU vocabulary, not a raw CRM
-- value.
SELECT
       D.[cr301_title]
      ,D.[cr301_doctorkey]
      ,D.[servhub_doctornameen]
      ,D.[cr301_doctornamear]
      ,D.[cr301_degreename]
      ,D.[cr301_specialtyname]
      ,D.[cr301_subspecialtyname]
      ,D.[cr18c_manualspecialtyname]
      ,D.[cr18c_manualsubspecialtyname]
      ,D.[cr301_contracttypename]
      ,D.[cr301_businessunitname]
      ,D.[cr18c_buname]
      ,D.[cr301_nationalityname]
      ,D.[statuscodename]
      ,D.[cr301_opdflag]
      ,D.[cr18c_exclusivenessname]
      ,D.[cr301_stardoctorname]
      ,D.[cr18c_firstpriorityname]
      ,D.[cr301_drnotes]
      ,D.[cr301_scopeofservice]
      ,D.[cr301_scopeofservicear]
      ,D.[cr301_qualificationsandexperience]
      ,D.[cr301_qualificationsandexperiencear]
      ,D.[servhub_examinationage]
      ,D.[cr603_insuranceissues]
      ,D.[cr603_staffexlimitation]
      ,D.[cr301_flag]
      ,F.[cr301_originalconsultationfees]
      ,F.[cr301_walkinconsultationfees]
      ,F.[cr301_contractconsultationfees]
      ,F.[cr301_affiliateconsultationfees]
FROM [dbo].[cr301_newdoctordataset] AS D
LEFT JOIN [dbo].[cr301_table1] AS F
       ON D.[cr301_doctorkey] = F.[cr301_doctorkey]
ORDER BY D.[cr301_doctorkey];
