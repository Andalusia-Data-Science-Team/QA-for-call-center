/****** FAQ Lookup Query ******/
-- This query retrieves FAQ records from Service_Hub database
-- Used for FAQ validation in the QA system
SELECT TOP (25000) 
    [ID],
    [Attachments],
    [AgentName],
    [BU],
    [Created_at],
    [CRM_x0020_BU],
    [Date],
    [Modified_BY],
    [FAQID],
    [Speciality],
    [End Call Result],
    [field5],
    [field8],
    [FQA],
    [Inquiry],
    [Injected to CRM],
    [mobile_phone],
    [Modified_at],
    [Response],
    [CST Inquiry / Request Details],
    [CustomerName]
FROM [Service_Hub].[dbo].[FAQ]
WHERE 
    (
        REPLACE(REPLACE(REPLACE([mobile_phone], ' ', ''), '-', ''), '+', '') = ?
        OR [mobile_phone] = ?
    )
    AND [Date] = ?
ORDER BY [Modified_at] DESC
