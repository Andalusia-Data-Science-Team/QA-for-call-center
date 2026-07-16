SELECT
    A.UniqueId AS AccountID,
    A.CreationDateTime AS AccountCreationDateTime,
    A.EmailAddress AS AccountEmailAddress,
    A.LastUpdatedDateTime AS AccountLastUpdatedDateTime,

    P.UniqueId AS PersonID,
    P.Avatar,
    P.CreationDateTime AS PersonCreationDateTime,
    P.Discriminator,
    P.EmailAddress AS PersonEmailAddress,
    P.ExternalIdentifier,
    P.FirstName,
    P.LastName,
    CONCAT(
        ISNULL(P.FirstName, ''),
        CASE
            WHEN P.FirstName IS NOT NULL AND P.LastName IS NOT NULL THEN ' '
            ELSE ''
        END,
        ISNULL(P.LastName, '')
    ) AS FullName,
    P.LastUpdatedDateTime AS PersonLastUpdatedDateTime,
    P.MaxChats,
    P.MaxWaitingChats,
    P.UserPresence,
    P.LastSentDateTime,
    P.ContactGroup_UniqueId,
    P.Account_UniqueID,
    P.UserEmailAddress
FROM [ROBINDWH.ROBINHQ.COM].[RHQ_Andalusia_Group].[dbo].[Accounts] A
INNER JOIN [ROBINDWH.ROBINHQ.COM].[RHQ_Andalusia_Group].[dbo].[People] P
    ON A.UniqueId = P.Account_UniqueID