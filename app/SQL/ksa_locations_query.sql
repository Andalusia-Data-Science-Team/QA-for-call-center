-- Authoritative CRM location reference data used to resolve the patient-requested branch.
-- The validator determines KSA applicability from cr301_country, not a guessed BU code.
SELECT [createdon], [modifiedon], [createdbyname], [modifiedbyname], [owneridname], [statecode], [statecodename], [statuscode], [statuscodename], [cr301_branchname], [cr301_area], [cr301_country], [cr301_description], [cr18c_region]
FROM [dbo].[cr301_andalusialocations];
