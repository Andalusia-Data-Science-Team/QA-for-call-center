/* ============================================================================
   Candidate discovery for the automated sampler — READ ONLY against CM_DWH.

   Replaces the automatic path's use of NEWFIX_CM_Chat_Search.sql, which caps
   at SELECT TOP (10) and so can never see a full interval's candidates.

   Returns one row per conversation handled by a human agent in the window,
   with no message content. Content joins are deferred to transcripts_by_ids.sql
   for the handful of conversations actually selected — this query runs every
   ten minutes across the whole roster, so it must stay cheap.

   ATTRIBUTION (decision D2)
   ─────────────────────────
   The responsible agent is the human who sent the LAST non-system message,
   resolved Messages.Sender_UniqueId -> Participations -> People with
   People.Discriminator = 'User' (the CM value for a human agent; 'Relation'
   is the patient and 'Bot' is automation). Their final message timestamp is
   HandlingTimeUtc and decides both the reporting day and the interval.

   This deliberately does NOT use Conversations.InboxOwnerPersonUniqueId, which
   is the *current* inbox owner and is wrong for transferred chats — it is the
   attribution the architecture explicitly rejects. The owner is still returned
   as InboxOwnerEmail so the two can be compared during validation.

   TIMEZONE
   ────────
   Every timestamp is returned RAW, as stored by CM_DWH, i.e. UTC. The
   DATEADD(HOUR, 3, ...) literals used elsewhere in this repository are a fixed
   +03:00 that is wrong for Africa/Cairo for half the year. Bounds come in as
   UTC bounds computed by app/automation/calendar.py; nothing here applies
   an offset. The caller formats validated datetime values as UTC literals
   before passing this query to OPENQUERY for remote execution.

   TIME BOUNDS (positional placeholders in this template)
       1  WindowStartUtc   inclusive   (Touched pre-filter)
       2  WindowEndUtc     exclusive   (Touched pre-filter)
       3  WindowStartUtc   inclusive   (handling-time filter)
       4  WindowEndUtc     exclusive   (handling-time filter)
   ============================================================================ */

/* ---------------------------------------------------------------------------
   PERFORMANCE NOTE — why this is driven from Messages, not Conversations.

   The first version of this query selected FROM Conversations and filtered on
   an OUTER APPLY's result (a bound predicate on lastHuman.HandlingTimeUtc).
   That forces
   SQL Server to evaluate the APPLY for EVERY conversation that has ever
   existed before it can discard any — across a linked server. In production
   that timed out and then killed the connection: 179 intervals failed with
   'Communication link failure' and not one job was ever reserved.

   Driving from Messages with a date predicate on an indexed column reduces the
   candidate set to conversations actually touched in the window first, and
   only then does the per-conversation work.
   --------------------------------------------------------------------------- */

;WITH Touched AS
(
    /* Conversations a human agent sent a message in during the window. Small,
       selective, and the only set worth doing per-conversation work on. */
    SELECT DISTINCT m.Conversation_UniqueId AS ConversationId
    FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m
    JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Participations] part
        ON m.Sender_UniqueId = part.UniqueId
    JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[People] p
        ON part.Person_UniqueId = p.UniqueId
    WHERE m.CreationDateTime >= ?
      AND m.CreationDateTime <  ?
      AND p.Discriminator = 'User'
      AND p.UserEmailAddress IS NOT NULL
      AND ISNULL(m.Discriminator, '') NOT LIKE '%SystemMessage'
)

SELECT
    c.UniqueId                                          AS ConversationId,

    /* ── Attributed agent: sender of the last human message ─────────────── */
    lastHuman.AgentEmail                                AS AgentEmail,
    lastHuman.AgentFullName                             AS AgentFullName,
    lastHuman.PersonUniqueId                            AS AgentPersonId,
    lastHuman.HandlingTimeUtc                           AS HandlingTimeUtc,

    /* ── Current inbox owner, for comparison only — NOT the attribution ─── */
    owner.UserEmailAddress                              AS InboxOwnerEmail,
    CASE
        WHEN owner.UserEmailAddress IS NOT NULL
         AND lastHuman.AgentEmail IS NOT NULL
         AND owner.UserEmailAddress <> lastHuman.AgentEmail
        THEN 1 ELSE 0
    END                                                 AS OwnerDiffersFromHandler,

    /* ── Completion signals. State and AnswerState are raw CM codes whose
         meaning must be confirmed before being used as an eligibility rule;
         ArchiveDateTime IS NOT NULL is the signal the sampler relies on. ── */
    c.State                                             AS ConversationState,
    c.AnswerState                                       AS AnswerState,
    c.ArchiveDateTime                                   AS ArchiveDateTimeUtc,
    c.StartDateTime                                     AS StartDateTimeUtc,
    c.LastUpdatedDateTime                               AS LastUpdatedUtc,

    counts.HumanMessageCount,
    counts.TotalMessageCount,

    CASE WHEN fw.ForwardedTime IS NOT NULL THEN 1 ELSE 0 END AS WasForwarded,

    ws.WebStoreName

FROM Touched t

INNER JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Conversations] c
    ON c.UniqueId = t.ConversationId

LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[People] owner
    ON c.InboxOwnerPersonUniqueId = owner.UniqueId

LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Webstores] ws
    ON c.WebStore_UniqueId = ws.UniqueId

/* Last human-agent message overall — the attribution anchor. Evaluated only
   for the Touched set, not for every conversation in the warehouse. */
OUTER APPLY
(
    SELECT TOP (1)
        p.UniqueId                                      AS PersonUniqueId,
        p.UserEmailAddress                              AS AgentEmail,
        LTRIM(RTRIM(CONCAT(ISNULL(p.FirstName, ''), ' ', ISNULL(p.LastName, '')))) AS AgentFullName,
        m.CreationDateTime                              AS HandlingTimeUtc
    FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m
    JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Participations] part
        ON m.Sender_UniqueId = part.UniqueId
    JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[People] p
        ON part.Person_UniqueId = p.UniqueId
    WHERE m.Conversation_UniqueId = c.UniqueId
      AND p.Discriminator = 'User'
      AND p.UserEmailAddress IS NOT NULL
      AND ISNULL(m.Discriminator, '') NOT LIKE '%SystemMessage'
    ORDER BY m.CreationDateTime DESC, m.UniqueId DESC
) lastHuman

OUTER APPLY
(
    SELECT
        COUNT(*)                                        AS TotalMessageCount,
        SUM(CASE WHEN p2.Discriminator = 'User' THEN 1 ELSE 0 END) AS HumanMessageCount
    FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m2
    LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Participations] part2
        ON m2.Sender_UniqueId = part2.UniqueId
    LEFT JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[People] p2
        ON part2.Person_UniqueId = p2.UniqueId
    WHERE m2.Conversation_UniqueId = c.UniqueId
) counts

OUTER APPLY
(
    SELECT TOP (1) m3.CreationDateTime AS ForwardedTime
    FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m3
    WHERE m3.Conversation_UniqueId = c.UniqueId
      AND m3.Discriminator = 'ConversationForwardedSystemMessage'
    ORDER BY m3.CreationDateTime ASC
) fw

WHERE
    /* The window applies to the HANDLING time, so a chat that began earlier but
       was last handled inside the interval belongs to that interval. */
    lastHuman.HandlingTimeUtc >= ?
    AND lastHuman.HandlingTimeUtc < ?

ORDER BY
    lastHuman.AgentEmail,
    lastHuman.HandlingTimeUtc,
    c.UniqueId;
