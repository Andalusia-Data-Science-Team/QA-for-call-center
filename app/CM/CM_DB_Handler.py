import pyodbc
import pandas as pd
import json
from pathlib import Path
import os
import re
from datetime import datetime

class CMDatabaseHandler:
    """Handler for connecting to SQL Server and executing queries"""
    
    @staticmethod
    def load_credentials_from_json(json_file, db_key):
        """
        Load database credentials from passcode.json file
        
        Args:
            json_file: Path to passcode.json file
            db_key: Key in DB_NAMES section (e.g., 'AHJ_DOT-CARE', 'BI')
            
        Returns:
            Dictionary with credentials (Server, Database, UID, PWD)
        """
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
            
            if 'DB_NAMES' in config and db_key in config['DB_NAMES']:
                credentials = config['DB_NAMES'][db_key]
                return {
                    'server': credentials.get('Server'),
                    'database': credentials.get('Database'),
                    'username': credentials.get('UID'),
                    'password': credentials.get('PWD'),
                    'driver': credentials.get('driver', '{ODBC Driver 17 for SQL Server}')
                }
            else:
                raise KeyError(f"Database key '{db_key}' not found in {json_file}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Passcode file not found: {json_file}")
        except Exception as e:
            raise Exception(f"Error loading credentials: {e}")
    
    def __init__(self, server=None, database=None, username=None, password=None, json_file=None, db_key=None):
        """
        Initialize database connection parameters
        
        Args:
            server: SQL Server instance (e.g., 'ROBINDWH.ROBINHQ.COM')
            database: Database name
            username: SQL authentication username (optional)
            password: SQL authentication password (optional)
            json_file: Path to passcode.json for loading credentials
            db_key: Database key in passcode.json (e.g., 'AHJ_DOT-CARE')
        """
        # Load from JSON if provided
        if json_file and db_key:
            creds = self.load_credentials_from_json(json_file, db_key)
            self.server = creds['server']
            self.database = creds['database']
            self.username = creds['username']
            self.password = creds['password']
            self.driver = creds['driver']
            print(f"✓ Credentials loaded from {json_file} for key '{db_key}'")
        else:
            self.server = server
            self.database = database
            self.username = username
            self.password = password
            self.driver = '{ODBC Driver 17 for SQL Server}'
        
        self.connection = None
    
    def connect(self):
        """Establish connection to SQL Server"""
        try:
            if self.username and self.password:
                # SQL Server Authentication
                connection_string = (
                    f'Driver={self.driver};'
                    f'Server={self.server};'
                    f'Database={self.database};'
                    f'UID={self.username};'
                    f'PWD={self.password};'
                )
            else:
                # Windows Authentication
                connection_string = (
                    f'Driver={self.driver};'
                    f'Server={self.server};'
                    f'Database={self.database};'
                    f'Trusted_Connection=yes;'
                )
            
            self.connection = pyodbc.connect(connection_string)
            print(f"✓ Connected to {self.server}.{self.database}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def execute_query_from_file(self, query_file, params=None):
        """
        Execute SQL query from a file and return results as DataFrame
        
        Args:
            query_file: Path to SQL file
            params: Dictionary of parameters for query substitution
            
        Returns:
            pandas DataFrame with query results
        """
        if not self.connection:
            raise ConnectionError("No active database connection")
        
        try:
            with open(query_file, 'r') as f:
                query = f.read()
            
            print(f"Executing query from: {query_file}")
            
            # Substitute parameters if provided
            if params:
                for key, value in params.items():
                    # Convert None to NULL string for SQL, otherwise wrap in quotes
                    if value is None:
                        sql_value = "NULL"
                    elif isinstance(value, (int, float)):
                        sql_value = str(value)
                    else:
                        # Escape single quotes in string values
                        sql_value = f"'{str(value).replace(chr(39), chr(39)+chr(39))}'"
                    
                    # Replace only :ParameterName patterns (not @ParameterName which are variable names)
                    query = re.sub(rf':\s*{re.escape(key)}\b', sql_value, query, flags=re.IGNORECASE)
                
                print(f"Parameters substituted: {list(params.keys())}")
            
            df = pd.read_sql(query, self.connection)
            print(f"✓ Query executed successfully. Retrieved {len(df)} rows.")
            return df
        except Exception as e:
            print(f"✗ Query execution failed: {e}")
            raise
    
    def save_to_excel(self, dataframe, output_file=None, sheet_name='Sheet1'):
        """
        Save DataFrame to Excel file
        
        Args:
            dataframe: pandas DataFrame to save
            output_file: Output Excel file path (auto-generated if not provided)
            sheet_name: Name of the Excel sheet
            
        Returns:
            Path to saved file
        """
        try:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"CM_Users_{timestamp}.xlsx"
            
            dataframe.to_excel(output_file, sheet_name=sheet_name, index=False)
            print(f"✓ Excel file saved: {output_file}")
            return output_file
        except Exception as e:
            print(f"✗ Failed to save Excel file: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✓ Database connection closed")


def main():
    """Main function to execute the workflow"""
    
    # Configuration
    SQL_FILE = "../SQL/CM_users.sql"
    OUTPUT_FILE = "CM_Users_Export.xlsx"
    
    # Method 1: Load credentials from passcode.json
    PASSCODE_FILE = "../passcode.json"  # Update path if needed
    DB_KEY = "ROBINDWH"  # Update to match your key in passcode.json
    
    # Method 2: Direct configuration
    DB_SERVER = "ROBINDWH.ROBINHQ.COM"
    DB_NAME = "RHQ_Andalusia_Group"
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    
    # Initialize handler - Choose one method:
    
    # Option 1: Load from passcode.json
    try:
        handler = CMDatabaseHandler(
            json_file=PASSCODE_FILE,
            db_key=DB_KEY
        )
    except Exception as e:
        print(f"Could not load from passcode.json: {e}")
        print("Falling back to direct configuration...")
        # Option 2: Use direct parameters
        handler = CMDatabaseHandler(
            server=DB_SERVER,
            database=DB_NAME,
            username=DB_USERNAME,
            password=DB_PASSWORD
        )
    
    try:
        # Connect to database
        if not handler.connect():
            return
        
        # Execute query and get results
        df = handler.execute_query_from_file(SQL_FILE)
        
        # Display preview
        print("\nData Preview:")
        print(df.head())
        print(f"\nTotal records: {len(df)}")
        
        # Save to Excel
        handler.save_to_excel(df, OUTPUT_FILE)
        
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        handler.close()


if __name__ == "__main__":
    main()