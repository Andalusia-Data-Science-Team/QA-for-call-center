-- Enhanced SQL Query with Overlap Strategy to Prevent Data Gaps
-- This query uses a 4.5-hour window instead of 4 hours to create overlap
-- and ensure no data is missed between runs

SELECT   
    V.ID AS visit_id,
    V.CreatedDate AS start_date,
    V.UpdatedDate AS end_date,
    P.ID AS patient_id,
    CONVERT(DATE,P.DateOfBirth) AS date_of_birth,
    P.OccupationID,
    OCC.EnName AS Occupation,
    CONCAT(P.EnFirstName, ' ', P.EnSecondName, ' ', P.EnThridName, ' ', P.EnLastName) AS patient_name,
    P.EnLastName AS family_name,
    P.EnFirstName AS pat_name_1,
    P.EnSecondName AS pat_name_2,
    CASE WHEN G.CODE='m' THEN 'male' else 'female' end AS gender,
    P.NationalityID,
    CASE 
        WHEN MS.CODE = 'm' THEN 'M'
        WHEN MS.CODE = 's' THEN 'S'
        WHEN MS.CODE = 'd' THEN 'D'
        WHEN MS.CODE = 'w' THEN 'W'
        WHEN MS.CODE = 'a' THEN 'A'
        WHEN MS.CODE = 'c Law' THEN 'C'
        WHEN MS.CODE = 'g' THEN 'G'
        WHEN MS.CODE = 'p' THEN 'P'
        WHEN MS.CODE = 'r' THEN 'R'
        WHEN MS.CODE = 'e' THEN 'E'
        WHEN MS.CODE = 'n' THEN 'N'
        WHEN MS.CODE = 'i' THEN 'I'
        WHEN MS.CODE = 'b' THEN 'B'
        WHEN MS.CODE = 'u' THEN 'U'
        WHEN MS.CODE = 'o' THEN 'O'
        WHEN MS.CODE = 't' THEN 'T'
        ELSE 'U'
    END AS marital_char,
    CASE P.IdentificationTypeID 
        WHEN 2 THEN 'NI'
        WHEN 3 THEN 'PPN'
        WHEN 5 THEN 'PRC'
        WHEN 8 THEN 'BORD'
        ELSE 'VISA' 
    END AS nationality,
    IDY.ENNAME,
    P.IdentificationValue AS iqama_no,
    1 AS Organization_Code,
    'ANDALUSIA HOSPITAL JEDDAH -  PROD' AS 'Organization Name',
    10000000046019 AS 'provider-license',
    vfi.ContractorClientPolicyNumber,
    vfi.ContractorCode AS insurer,
    vfi.InsuranceCardNo,
    vfi.ContractorClientEnName,
    vfi.ContractorClientID,
    CGWM.EnName AS purchaser_name,
    CGWM.Code AS payer_linces,
    -- Add metadata for tracking
    V.CreatedDate AS original_created_date,
    V.UpdatedDate AS original_updated_date,
    GETDATE() AS extraction_timestamp
FROM VisitMgt.Visit AS v
LEFT JOIN VisitMgt.VisitFinincailInfo AS vfi ON vfi.VisitID = v.id
LEFT JOIN MPI.Patient P ON p.id = v.patientid
LEFT JOIN MPI.SLKP_Occupation OCC ON P.OccupationID = OCC.ID
LEFT JOIN MPI.SLKP_Gender G ON P.GenderID = G.ID
LEFT JOIN MPI.SLKP_MaritalStatus MS ON P.MaritalStatusID = MS.ID
LEFT JOIN Billing.Contractor BC ON BC.ID = vfi.ContractorID 
LEFT JOIN MPI.LKP_IdentificationType IDY ON IDY.ID = P.IdentificationTypeID
INNER JOIN Billing.ContractorGateWayMappings CGWM ON CGWM.ContractorID = ISNULL(BC.ParentID, BC.ID) AND CGWM.GateWayID = 3
WHERE V.VisitStatusID != 3 
    AND V.FinancialStatusID = 2
    AND CONVERT(DATE, V.CreatedDate) >= DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0) -- from first of month
    AND CONVERT(DATE, V.CreatedDate) = CONVERT(DATE, GETDATE()) -- for today only
    -- Enhanced time window with 30-minute overlap to prevent gaps
    AND CONVERT(DATETIME, V.CreatedDate) >= DATEADD(MINUTE, -270, GETDATE()) -- 4.5 hours instead of 4
ORDER BY V.CreatedDate ASC; -- Ensure consistent ordering