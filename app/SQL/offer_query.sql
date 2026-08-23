SELECT
     [new_offerid]                   AS Offer_ID
    ,cr4be_dotcarecodenname          AS Offer_Name_EN
    ,[new_offernamear]               AS Offer_Name_AR
    ,[new_offerdescriptionen]        AS Offer_Description_EN
    ,[new_offerdescriptionar]        AS Offer_Description_AR
    ,[new_buname]                    AS BU
    ,[new_specialtyname]             AS Specialty
    ,[new_offerstatusnewname]        AS Offer_Status
    ,[new_publicationstatusname]     AS Publication_Status
    ,[new_totalofferbeforediscf]     AS Price_Before_Discount
    ,[new_totalofferafterdiscf]      AS Price_After_Discount
    ,[new_offerdiscountf]            AS Discount
    ,[new_startdatef]                AS Start_Date
    ,[new_offerenddatef]             AS End_Date
    ,[new_offertypef]                AS Offer_Type
    ,[new_offertagname]              AS Offer_Tag
    ,[new_servicecodedataset]        AS DotCare_Code
    ,[createdon]                     AS Created_On
FROM new_offer_equest
ORDER BY [createdon] DESC