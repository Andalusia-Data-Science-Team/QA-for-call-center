/* ============================================================================
   Transcript retrieval by explicit conversation id — READ ONLY against CM_DWH.

   Two things this fixes relative to NEWFIX_CM_Chat_Search.sql:

   1. NO 'SELECT TOP (10)'. The caller names exactly which conversations it
      wants, so there is no implicit cap and no silent loss.

   2. MESSAGE DEDUPLICATION. The joins to Participations, People, ChannelAccounts
      and Channels can each fan out and emit a message more than once. The
      ROW_NUMBER filter below keeps one row per (conversation, message), with a
      deterministic tie-break, so a duplicated join cannot inflate a transcript.

   The output column names match NEWFIX_CM_Chat_Search.sql exactly, so
   CallTranscript.from_cm_rows consumes the rows unchanged.

   TIMEZONE: unlike NEWFIX, this returns UTC and does not add three hours.
   from_cm_rows only derives call_date and a display HH:MM:SS from these, and
   the sampler has already decided the reporting day from the handling time.
   Passing a fixed +3 through would misdate conversations in the first hours
   of a Cairo winter day.

   PARAMETERS: the placeholder token below is replaced by the caller with the
   right number of question marks — the IDS THEMSELVES ARE ALWAYS BOUND, never
   interpolated. The token appears exactly twice, once per CTE, and the caller
   binds the id list twice in that same order. See app/CM/candidate_queries.py.
   ============================================================================ */

;WITH Targeted AS
(
    SELECT
        c.UniqueId,
        owner.UserEmailAddress AS EmailAddress,
        LTRIM(RTRIM(CONCAT(ISNULL(owner.FirstName, ''), ' ', ISNULL(owner.LastName, '')))) AS AgentFullName,
        c.StartDateTime         AS Start_DateTime,
        c.AnswerDateTime        AS Answer_DateTime,
        c.LastUpdatedDateTime   AS LastUpdatedDateTime,
        c.ArchiveDateTime       AS Archive_DateTime,
        fr.FirstResponse,
        fw.ForwardedTime,
        DATEDIFF(MINUTE, c.StartDateTime, fr.FirstResponse) AS First_Response_Minutes,
        c.InboxExternalIdentifier AS PatientPhoneNumber,
        c.AnswerState,
        c.State,
        'https://app.robinhq.com/#/conversation:' + CAST(c.UniqueId AS VARCHAR(100)) AS ConversationLink,
        ws.WebStoreName
    FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Conversations] c
    LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[People] owner
        ON c.InboxOwnerPersonUniqueId = owner.UniqueId
    LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Webstores] ws
        ON c.WebStore_UniqueId = ws.UniqueId
    OUTER APPLY
    (
        SELECT TOP (1) m2.CreationDateTime AS FirstResponse
        FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m2
        WHERE m2.Conversation_UniqueId = c.UniqueId
          AND m2.SenderWasOwner = 1
        ORDER BY m2.CreationDateTime ASC
    ) fr
    OUTER APPLY
    (
        SELECT TOP (1) m3.CreationDateTime AS ForwardedTime
        FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m3
        WHERE m3.Conversation_UniqueId = c.UniqueId
          AND m3.Discriminator = 'ConversationForwardedSystemMessage'
        ORDER BY m3.CreationDateTime ASC
    ) fw
    WHERE c.UniqueId IN ({id_placeholders})
),

/* One row per (conversation, message). Everything the message rows need is
   resolved here so the outer SELECT joins nothing that could fan out again. */
DedupedMessages AS
(
    SELECT
        m.Conversation_UniqueId,
        m.UniqueId              AS MessageId,
        m.SenderWasOwner,
        m.Content,
        m.CreationDateTime,
        m.Scope,
        ch.ChannelName,
        sender.UniqueId         AS SenderPersonId,
        sender.Discriminator    AS SenderType,
        LTRIM(RTRIM(CONCAT(ISNULL(sender.FirstName, ''), ' ', ISNULL(sender.LastName, '')))) AS SenderFullName,
        senderPart.UniqueId     AS RelationId,
        ROW_NUMBER() OVER (
            PARTITION BY m.Conversation_UniqueId, m.UniqueId
            ORDER BY ch.ChannelName, sender.UniqueId, senderPart.UniqueId
        ) AS rn
    FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m
    LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Participations] senderPart
        ON m.Sender_UniqueId = senderPart.UniqueId
    LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[People] sender
        ON senderPart.Person_UniqueId = sender.UniqueId
    LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[ChannelAccounts] ca
        ON m.ChannelAccount_UniqueId = ca.UniqueId
    LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Channels] ch
        ON ca.Channel_Id = ch.Id
    WHERE m.Conversation_UniqueId IN ({id_placeholders})
)

SELECT
    t.*,
    d.MessageId,
    d.SenderWasOwner,
    d.Content,
    d.CreationDateTime,
    d.ChannelName           AS ConversationChannel,
    d.Scope,
    d.SenderPersonId        AS OwnerId,
    d.SenderFullName        AS OwnerName,
    d.RelationId,
    d.SenderFullName        AS RelationName,
    d.SenderFullName        AS SenderIdName,
    d.SenderType,
    t.State                 AS ConversationState,
    t.Start_DateTime        AS ConversationCreationDateTime,
    t.ConversationLink      AS ConversationReferrer,
    d.ChannelName           AS Channel,
    t.AnswerState           AS ServiceLevelOnTime,
    CASE WHEN t.AnswerState IS NOT NULL THEN 1 ELSE 0 END AS IsAnswered,
    t.Archive_DateTime      AS ArchiveDateTime
FROM Targeted t
INNER JOIN DedupedMessages d
    ON d.Conversation_UniqueId = t.UniqueId
   AND d.rn = 1
ORDER BY
    t.UniqueId,
    d.CreationDateTime,
    d.MessageId;
