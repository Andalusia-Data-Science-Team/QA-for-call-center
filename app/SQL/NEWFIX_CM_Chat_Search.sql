DECLARE @ConversationId UNIQUEIDENTIFIER = NULL --:ConversationId ;
DECLARE @AgentFullName  NVARCHAR(200) = NULL --:AgentFullName;
DECLARE @AgentEmail     NVARCHAR(200) = 'aya.mansour@andalusiagroup.net'--:AgentEmail;
DECLARE @FilterDate     DATE = '2026-09-17' --:FilterDate;


;WITH ConvSummary AS
(
    SELECT
        c.UniqueId,

        owner.UserEmailAddress AS UserEmailAddress,

        LTRIM(
            RTRIM(
                CONCAT(
                    ISNULL(owner.FirstName, ''),
                    ' ',
                    ISNULL(owner.LastName, '')
                )
            )
        ) AS AgentFullName,

        DATEADD(HOUR, 3, c.StartDateTime)       AS Start_DateTime,
        DATEADD(HOUR, 3, c.AnswerDateTime)      AS Answer_DateTime,
        DATEADD(HOUR, 3, c.LastUpdatedDateTime) AS LastUpdatedDateTime,
        DATEADD(HOUR, 3, c.ArchiveDateTime)     AS Archive_DateTime,

        fr.FirstResponse,
        fw.ForwardedTime,

        DATEDIFF(
            MINUTE,
            DATEADD(HOUR, 3, c.StartDateTime),
            fr.FirstResponse
        ) AS First_Response_Minutes,

        c.InboxExternalIdentifier AS PatientPhoneNumber,

        c.AnswerState,
        c.State,

        'https://app.robinhq.com/#/conversation:'
            + CAST(c.UniqueId AS VARCHAR(100))
            AS ConversationLink,

        --c.TagName1 AS Tag1,
        --c.TagName2 AS Tag2,
        --c.TagName3 AS Tag3,
        --c.TagName4 AS Tag4,
        --c.TagName5 AS Tag5,

        ws.WebStoreName

    FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Conversations] c

    LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[People] owner
        ON c.InboxOwnerPersonUniqueId = owner.UniqueId

    LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Webstores] ws
        ON c.WebStore_UniqueId = ws.UniqueId

    OUTER APPLY
    (
        SELECT TOP (1)
            DATEADD(HOUR, 3, m2.CreationDateTime) AS FirstResponse

        FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m2

        WHERE
            m2.Conversation_UniqueId = c.UniqueId
            AND m2.SenderWasOwner = 1

        ORDER BY
            m2.CreationDateTime ASC
    ) fr

    OUTER APPLY
    (
        SELECT TOP (1)
            DATEADD(HOUR, 3, m3.CreationDateTime) AS ForwardedTime

        FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m3

        WHERE
            m3.Conversation_UniqueId = c.UniqueId
            AND m3.Discriminator = 'ConversationForwardedSystemMessage'

        ORDER BY
            m3.CreationDateTime ASC
    ) fw

    WHERE
        (
            @ConversationId IS NULL
            OR c.UniqueId = @ConversationId
        )

        AND
        (
            @AgentFullName IS NULL
            OR LTRIM(
                RTRIM(
                    CONCAT(
                        ISNULL(owner.FirstName, ''),
                        ' ',
                        ISNULL(owner.LastName, '')
                    )
                )
            ) LIKE '%' + @AgentFullName + '%'
        )

        AND
        (
            @AgentEmail IS NULL
            OR owner.UserEmailAddress LIKE '%' + @AgentEmail + '%'
        )

        AND
        (
            @FilterDate IS NULL

            OR
            (
                c.StartDateTime >=
                    DATEADD(
                        HOUR,
                        -3,
                        CAST(@FilterDate AS DATETIME)
                    )

                AND c.StartDateTime <
                    DATEADD(
                        HOUR,
                        -3,
                        DATEADD(
                            DAY,
                            1,
                            CAST(@FilterDate AS DATETIME)
                        )
                    )
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

    /* Same naming style as your reference query */

    m.SenderWasOwner,

    m.Content,

    DATEADD(
        HOUR,
        3,
        m.CreationDateTime
    ) AS CreationDateTime,

    /* Corresponds to your old ConversationChannel */
    ch.ChannelName AS ConversationChannel,

    /* Raw integer in CM DWH */
    m.Scope,

    /* Current owner/person linked to the message sender */
    sender.UniqueId AS OwnerId,

    LTRIM(
        RTRIM(
            CONCAT(
                ISNULL(sender.FirstName, ''),
                ' ',
                ISNULL(sender.LastName, '')
            )
        )
    ) AS OwnerName,

    /* Participation identifier */
    senderPart.UniqueId AS RelationId,

    LTRIM(
        RTRIM(
            CONCAT(
                ISNULL(sender.FirstName, ''),
                ' ',
                ISNULL(sender.LastName, '')
            )
        )
    ) AS RelationName,

    LTRIM(
        RTRIM(
            CONCAT(
                ISNULL(sender.FirstName, ''),
                ' ',
                ISNULL(sender.LastName, '')
            )
        )
    ) AS SenderIdName,

    sender.Discriminator AS SenderType,

    tc.State AS ConversationState,

    tc.Start_DateTime AS ConversationCreationDateTime,

    tc.ConversationLink AS ConversationReferrer,

    ch.ChannelName AS Channel,

    tc.AnswerState AS ServiceLevelOnTime,

    CASE
        WHEN tc.AnswerState IS NOT NULL THEN 1
        ELSE 0
    END AS IsAnswered,

    tc.Archive_DateTime AS ArchiveDateTime,

    tc.WebStoreName

FROM TopConversations tc

INNER JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m
    ON m.Conversation_UniqueId = tc.UniqueId

LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Participations] senderPart
    ON m.Sender_UniqueId = senderPart.UniqueId

LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[People] sender
    ON senderPart.Person_UniqueId = sender.UniqueId

LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[ChannelAccounts] ca
    ON m.ChannelAccount_UniqueId = ca.UniqueId

LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Channels] ch
    ON ca.Channel_Id = ch.Id

ORDER BY
    tc.Start_DateTime DESC,
    tc.UniqueId,
    m.CreationDateTime;