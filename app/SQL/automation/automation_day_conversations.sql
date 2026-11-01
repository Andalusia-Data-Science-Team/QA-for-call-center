/* ============================================================================
   Automation candidate discovery — NEWFIX-oriented. READ ONLY against CM_DWH.

   Built from NEWFIX_CM_Chat_Search.sql's ConvSummary: the same Conversations
   table, the same inbox-owner join for the agent, the same Webstores join.
   It returns one row per conversation and NO message content — content is
   fetched only for the conversations actually selected, by
   automation_chat_by_ids.sql.

   ATTRIBUTION: the agent is the conversation's inbox owner,
   owner.UserEmailAddress — exactly how NEWFIX and the manual dashboard
   attribute a chat to an agent.

   WINDOW: ArchiveDateTime, not StartDateTime. A conversation belongs to the
   interval in which it was closed. That guarantees every candidate is already
   finished when its interval is snapshotted; windowing on StartDateTime would
   capture chats that are still open at snapshot time, mark them ineligible,
   and never look at them again.

   AgentReplied is 1 when the owner sent at least one message
   (SenderWasOwner = 1, the same test NEWFIX uses for FirstResponse). A closed
   conversation the owner never replied to has no agent performance to assess.

   PERFORMANCE: every predicate is a direct column comparison on Conversations,
   so the linked server filters before anything else happens. Measured at about
   6 seconds per 2-hour window on 2026-10-18. There are no correlated
   OUTER APPLYs here — those ran row-by-row across the linked server and timed
   out the previous candidate query.

   TIMEZONE: raw stored values, which are UTC. The window bounds are UTC values
   computed by app/automation/calendar.py, so Africa/Cairo's DST changes are
   handled outside this file and no fixed offset is applied here.

   PARAMETERS (positional, bound by the driver)
       1  WindowStartUtc   inclusive
       2  WindowEndUtc     exclusive
   ============================================================================ */

SELECT
    c.UniqueId                                          AS ConversationId,

    owner.UserEmailAddress                              AS AgentEmail,
    LTRIM(RTRIM(CONCAT(ISNULL(owner.FirstName, ''), ' ', ISNULL(owner.LastName, ''))))
                                                        AS AgentFullName,

    c.ArchiveDateTime                                   AS HandlingTimeUtc,
    c.ArchiveDateTime                                   AS ArchiveDateTimeUtc,
    c.StartDateTime                                     AS StartDateTimeUtc,
    c.LastUpdatedDateTime                               AS LastUpdatedUtc,

    c.State                                             AS ConversationState,
    c.AnswerState                                       AS AnswerState,

    CASE
        WHEN EXISTS (
            SELECT 1
            FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m
            WHERE m.Conversation_UniqueId = c.UniqueId
              AND m.SenderWasOwner = 1
        )
        THEN 1 ELSE 0
    END                                                 AS AgentReplied,

    ws.WebStoreName

FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Conversations] c

LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[People] owner
    ON c.InboxOwnerPersonUniqueId = owner.UniqueId

LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Webstores] ws
    ON c.WebStore_UniqueId = ws.UniqueId

WHERE
    c.ArchiveDateTime >= ?
    AND c.ArchiveDateTime < ?

ORDER BY
    owner.UserEmailAddress,
    c.ArchiveDateTime,
    c.UniqueId;
