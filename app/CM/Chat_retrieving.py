from CM_DB_Handler import CMDatabaseHandler

try:
    # Initialize database handler with credentials from passcode.json
    db_handler = CMDatabaseHandler(json_file="passcode.json", db_key="CM")
    
    # Connect to database
    if db_handler.connect():
        print("Database connection established successfully.")
    else:
        print("Failed to connect to database.")

    chats = db_handler.execute_query_from_file("../SQL/CM_Chat_Search.sql", params={
        "ConversationId": None,
        "AgentFullName": None,
        "AgentEmail": "Nada.Abdullah@Andalusiagroup.net",
        "FilterDate": "2026-06-05"
    })
    print(f"✓ Query executed successfully")
    # Display retrieved chats
    if chats is not None:
        print(f"\nRetrieved {len(chats)} chat conversations:")
        print(chats['UniqueId'].unique())
    else:
        print("No chats retrieved.")

finally:
    # Close connection when done
    db_handler.close()