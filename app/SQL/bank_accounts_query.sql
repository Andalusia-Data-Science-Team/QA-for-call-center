-- Authoritative CRM bank-account reference data for deterministic KSA QA validation.
-- Status/BU semantics are interpreted in Python only when tenant evidence exists.
SELECT [createdon], [modifiedon], [createdbyname], [modifiedbyname], [owneridname], [owningbusinessunitname], [statecode], [statecodename], [statuscode], [statuscodename], [importsequencenumber], [cr301_accountnumber], [cr301_accountowner], [cr301_bankname], [cr301_bu], [cr301_ibannumber], [cr301_notes], [servhub_bulookupname], [cr18c_bunname]
FROM [dbo].[cr301_bankaccounts];
