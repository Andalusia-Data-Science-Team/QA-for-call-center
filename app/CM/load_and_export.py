#!/usr/bin/env python3
"""
Script to load SQL query and export results to Excel
Supports reading credentials from passcode.json file
"""

import pyodbc
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import os
import sys

# Configuration
PASSCODE_FILE = "/home/ai/Workspace/Rafik/QA_System-main/app/Passcode.json"  # Path to credentials file
DB_KEY = "CM"  # Key in passcode.json DB_NAMES section

# Fallback configuration (if passcode.json not found)
DB_SERVER = "ROBINDWH.ROBINHQ.COM"
DB_NAME = "RHQ_Andalusia_Group"
SQL_FILE = "/home/ai/Workspace/Rafik/QA_System-main/app/SQL/CM_users.sql"
OUTPUT_FILE = f"CM_Users.xlsx"

# Optional: Override with environment variables
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def load_credentials_from_json(json_file, db_key):
    """Load database credentials from passcode.json"""
    try:
        with open(json_file, 'r') as f:
            config = json.load(f)
        
        if 'DB_NAMES' in config and db_key in config['DB_NAMES']:
            creds = config['DB_NAMES'][db_key]
            return {
                'server': creds.get('Server'),
                'database': creds.get('Database'),
                'username': creds.get('UID'),
                'password': creds.get('PWD'),
                'driver': creds.get('driver', '{ODBC Driver 17 for SQL Server}')
            }
        else:
            print(f"✗ Database key '{db_key}' not found in DB_NAMES")
            return None
    except FileNotFoundError:
        print(f"✗ Passcode file not found: {json_file}")
        return None
    except json.JSONDecodeError:
        print(f"✗ Invalid JSON in {json_file}")
        return None
    except Exception as e:
        print(f"✗ Error loading credentials: {e}")
        return None


def connect_to_database(server, database, username, password, driver='{ODBC Driver 17 for SQL Server}'):
    """Establish connection to SQL Server"""
    try:
        if username and password:
            # SQL Server Authentication
            connection_string = (
                f'Driver={driver};'
                f'Server={server};'
                f'Database={database};'
                f'UID={username};'
                f'PWD={password};'
            )
        else:
            # Windows Authentication
            connection_string = (
                f'Driver={driver};'
                f'Server={server};'
                f'Database={database};'
                f'Trusted_Connection=yes;'
            )
        
        connection = pyodbc.connect(connection_string)
        print(f"✓ Connected to {server}.{database}")
        return connection
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        sys.exit(1)


def execute_query(connection, query_file):
    """Execute SQL query from file"""
    try:
        if not Path(query_file).exists():
            print(f"✗ Query file not found: {query_file}")
            sys.exit(1)
        
        with open(query_file, 'r') as f:
            query = f.read()
        
        print(f"Executing query from: {query_file}")
        df = pd.read_sql(query, connection)
        print(f"✓ Query executed successfully. Retrieved {len(df)} rows.")
        return df
    except Exception as e:
        print(f"✗ Query execution failed: {e}")
        sys.exit(1)


def export_to_excel(dataframe, output_file):
    """Export DataFrame to Excel"""
    try:
        dataframe.to_excel(output_file, sheet_name='CM_Users', index=False)
        print(f"✓ Excel file saved: {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("EXPORT SUMMARY")
        print("="*50)
        print(f"Total Records: {len(dataframe)}")
        print(f"Columns: {', '.join(dataframe.columns)}")
        print(f"Output File: {output_file}")
        print("="*50)
        
        return output_file
    except Exception as e:
        print(f"✗ Failed to save Excel file: {e}")
        sys.exit(1)


def main():
    """Main execution"""
    print("Loading CM Users Query and Exporting to Excel")
    print("="*50)
    
    # Try to load credentials from passcode.json
    server = None
    database = None
    username = None
    password = None
    driver = '{ODBC Driver 17 for SQL Server}'
    
    creds = load_credentials_from_json(PASSCODE_FILE, DB_KEY)
    if creds:
        server = creds['server']
        database = creds['database']
        username = creds['username']
        password = creds['password']
        driver = creds['driver']
        print(f"✓ Credentials loaded from {PASSCODE_FILE}")
    else:
        # Fall back to direct configuration
        print("Using default configuration...")
        server = DB_SERVER
        database = DB_NAME
        username = DB_USERNAME
        password = DB_PASSWORD
    
    # Connect
    connection = connect_to_database(server, database, username, password, driver)
    
    try:
        # Execute query
        df = execute_query(connection, SQL_FILE)
        
        # Display preview
        print("\nData Preview (first 5 rows):")
        print(df.head().to_string())
        
        # Export to Excel
        export_to_excel(df, OUTPUT_FILE)
        
    finally:
        connection.close()
        print("✓ Database connection closed")


if __name__ == "__main__":
    main()
