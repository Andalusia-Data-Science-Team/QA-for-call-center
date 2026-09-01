DECLARE @ConversationId UNIQUEIDENTIFIER = :ConversationId ;
DECLARE @AgentFullName  NVARCHAR(200) = :AgentFullName;
DECLARE @AgentEmail     NVARCHAR(200) = :AgentEmail;
DECLARE @FilterDate     DATE = :FilterDate;

;WITH ConvSummary AS
(
    -- Your existing CTE exactly as it is
    SELECT
        c.UniqueId,
        a.EmailAddress,
        p.FirstName + ' ' + p.LastName AS AgentFullName,
        DATEADD(HOUR, 3, c.StartDateTime)       AS Start_DateTime,
        DATEADD(HOUR, 3, c.AnswerDateTime)      AS Answer_DateTime,
        DATEADD(HOUR, 3, c.LastUpdatedDateTime) AS LastUpdatedDateTime,
        DATEADD(HOUR, 3, c.ArchiveDateTime)     AS Archive_DateTime,
        fr.FirstResponse,
        fw.ForwardedTime,
        DATEDIFF(MINUTE, DATEADD(HOUR, 3, c.StartDateTime), fr.FirstResponse) AS First_Response_Minutes,
        c.InboxExternalIdentifier AS PatientPhoneNumber,
        c.AnswerState,
        c.State,
        'https://app.robinhq.com/#/conversation:' + CAST(c.UniqueId AS VARCHAR(100)) AS ConversationLink,
        c.TagName1 AS Tag1,
        c.TagName2 AS Tag2,
        c.TagName3 AS Tag3,
        c.TagName4 AS Tag4,
        c.TagName5 AS Tag5
    FROM [ROBINDWH.ROBINHQ.COM].[RHQ_Andalusia_Group].[dbo].[Conversations] c
    LEFT JOIN [ROBINDWH.ROBINHQ.COM].[RHQ_Andalusia_Group].[dbo].[People] p
        ON c.ArchivePersonUniqueId = p.UniqueID
    LEFT JOIN [ROBINDWH.ROBINHQ.COM].[RHQ_Andalusia_Group].[dbo].[Accounts] a
        ON p.Account_UniqueID = a.UniqueID
    OUTER APPLY
    (
        SELECT TOP (1)
            DATEADD(HOUR, 3, m2.CreationDateTime) AS FirstResponse
        FROM [ROBINDWH.ROBINHQ.COM].[RHQ_Andalusia_Group].[dbo].[Messages] m2
        WHERE m2.Conversation_UniqueId = c.UniqueId
          AND m2.CollaborationMode = 'Internal'
        ORDER BY m2.CreationDateTime
    ) fr
    OUTER APPLY
    (
        SELECT TOP (1)
            DATEADD(HOUR, 3, m3.CreationDateTime) AS ForwardedTime
        FROM [ROBINDWH.ROBINHQ.COM].[RHQ_Andalusia_Group].[dbo].[Messages] m3
        WHERE m3.Conversation_UniqueId = c.UniqueId
          AND m3.Discriminator = 'ConversationForwardedSystemMessage'
        ORDER BY m3.CreationDateTime
    ) fw
    WHERE
        (@ConversationId IS NULL OR c.UniqueId = @ConversationId)
        AND (@AgentFullName IS NULL OR (p.FirstName + ' ' + p.LastName) LIKE '%' + @AgentFullName + '%')
        AND (@AgentEmail IS NULL OR a.EmailAddress LIKE '%' + @AgentEmail + '%')
        AND (
            @FilterDate IS NULL
            OR (
                c.StartDateTime >= DATEADD(HOUR, -3, CAST(@FilterDate AS DATETIME))
                AND c.StartDateTime < DATEADD(HOUR, -3, DATEADD(DAY, 1, CAST(@FilterDate AS DATETIME)))
            )
        )
),
TopConversations AS
(
    SELECT TOP (10) *
    FROM ConvSummary
    ORDER BY Start_DateTime DESC
)
SELECT
    tc.*,
    --m.[MessageId],
    --m.[SenderId],
    m.[SenderWasOwner],
    m.[Content],
    m.[CreationDateTime],
    m.[ConversationChannel],
    m.[Scope],
    --m.[Rating],
    m.[OwnerId],
    m.[OwnerName],
    m.[RelationId],
    m.[RelationName],
    m.[SenderIdName],
    m.[SenderType],
    --m.[WebStoreId],
    m.[ConversationState],
    m.[ConversationCreationDateTime],
    m.[ConversationReferrer],
    m.[Channel],
    m.[ServiceLevelOnTime],
    m.[IsAnswered],
    --m.[Category],
    --m.[LinkedOrderNumbers],
    --m.[Subject],
    m.[ArchiveDateTime],
    m.[WebStoreName]
FROM TopConversations tc
INNER JOIN [ROBINDWH.ROBINHQ.COM].[RHQ_Andalusia_Group].[dbo].[MessagesTotal] m
    ON m.ConversationId = tc.UniqueId
--where m.ConversationId = '499DBB06-5279-F111-B337-000D3AA9D4A7'
ORDER BY
    tc.Start_DateTime DESC,
    tc.UniqueId,
    m.CreationDateTime;
