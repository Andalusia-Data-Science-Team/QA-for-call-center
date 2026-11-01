/* ============================================================================
   Automation roster — the agents who actually handle chats in CM.
   READ ONLY against CM_DWH.

   WHY NOT DWH.AI.Agents
   ─────────────────────
   On 2026-10-17, 18 people owned conversations in CM. Only 3 of them were in
   DWH.AI.Agents; the busiest CM agents were not on that list at all. Planning
   against it gave quota to 46 people of whom 43 had no chats, and the day
   finished at 12 of 300. The manual dashboard already takes its agent list
   from CM (CM_users.sql) and NEWFIX attributes chats by inbox owner, so the
   roster is taken from the same place.

   The roster is everyone who owned a closed conversation they replied to
   during a look-back window that ENDS where the reporting day starts — so it
   is fixed before the day begins and is not shaped by that day's traffic.
   Requiring at least one reply leaves out system and admin accounts that are
   only ever auto-assigned conversations.

   PARAMETERS (positional, bound by the driver)
       1  LookbackStartUtc   inclusive
       2  LookbackEndUtc     exclusive (the reporting day's start)
   ============================================================================ */

SELECT
    owner.UserEmailAddress                              AS Agent_Email_Address,
    MAX(LTRIM(RTRIM(CONCAT(ISNULL(owner.FirstName, ''), ' ', ISNULL(owner.LastName, '')))))
                                                        AS Agent_Name,
    CAST(owner.UniqueId AS VARCHAR(100))                AS Agent_id,
    COUNT(*)                                            AS ConversationCount

FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Conversations] c

INNER JOIN [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[People] owner
    ON c.InboxOwnerPersonUniqueId = owner.UniqueId

WHERE
    c.ArchiveDateTime >= ?
    AND c.ArchiveDateTime < ?
    AND owner.UserEmailAddress IS NOT NULL
    AND EXISTS (
        SELECT 1
        FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Messages] m
        WHERE m.Conversation_UniqueId = c.UniqueId
          AND m.SenderWasOwner = 1
    )

GROUP BY
    owner.UserEmailAddress,
    owner.UniqueId

ORDER BY
    owner.UserEmailAddress;
