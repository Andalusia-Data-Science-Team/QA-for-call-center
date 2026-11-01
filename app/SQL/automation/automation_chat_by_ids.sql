/* ============================================================================
   Automation copy of NEWFIX_CM_Chat_Search.sql — transcripts by conversation id.
   READ ONLY against CM_DWH. NEWFIX_CM_Chat_Search.sql itself is NOT modified;
   the manual dashboard keeps using it unchanged.

   Identical to NEWFIX — same CTEs, joins, columns, column names and the same
   DATEADD(HOUR, 3, ...) presentation of timestamps, so CallTranscript.from_cm_rows
   builds exactly the transcript the manual dashboard builds, and automated rows
   in DWH.AI.Call_QA_Results carry the same Call_Date convention as manual ones —
   except for three things the pipeline needs:

     1. Selection is an explicit list of conversation ids instead of the
        agent / name / date filters. The token below is replaced with one bound
        placeholder per id by app/CM/candidate_queries.py; ids are never spliced in.
     2. No TOP: the caller names exactly the conversations it wants, so nothing
        can be silently cut off.
     3. m.UniqueId is returned as MessageId and used as the final ORDER BY key,
        so duplicated join rows can be removed and messages sharing a timestamp
        come back in a stable order.
   ============================================================================ */

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
        c.UniqueId IN ({id_placeholders})
),

TopConversations AS
(
    SELECT *
    FROM ConvSummary
)

SELECT
    tc.*,

    /* Same naming style as your reference query */

    m.UniqueId AS MessageId,

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
    m.CreationDateTime,
    m.UniqueId;