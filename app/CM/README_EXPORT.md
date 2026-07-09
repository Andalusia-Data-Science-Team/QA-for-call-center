# CM Users Export Script

This solution provides two ways to load the SQL query and export results to Excel, with support for reading credentials from a `passcode.json` file.

## Files

1. **CM_DB_Handler.py** - Reusable database handler class
2. **load_and_export.py** - Standalone script (recommended for quick use)
3. **passcode.json** - Database credentials file (template provided)

## Prerequisites

Install required packages:
```bash
pip install pyodbc pandas openpyxl
```

On Linux, you may also need:
```bash
sudo apt-get install unixodbc unixodbc-dev
```

## Configuration

### Option 1: Using passcode.json (Recommended)

1. Edit `passcode.json` and add your credentials:
```json
{
    "DB_NAMES": {
        "ROBINDWH": {
            "Server": "ROBINDWH.ROBINHQ.COM",
            "Database": "RHQ_Andalusia_Group",
            "UID": "your_username",
            "PWD": "your_password",
            "driver": "{ODBC Driver 17 for SQL Server}"
        }
    }
}
```

2. Run the script:
```bash
python load_and_export.py
```

### Option 2: Using Environment Variables
```bash
export DB_USERNAME="your_username"
export DB_PASSWORD="your_password"
python load_and_export.py
```

### Option 3: Using the Handler Class with passcode.json
```python
from CM_DB_Handler import CMDatabaseHandler

handler = CMDatabaseHandler(
    json_file="passcode.json",
    db_key="ROBINDWH"
)

handler.connect()
df = handler.execute_query_from_file("SQL/CM_users.sql")
handler.save_to_excel(df, "output.xlsx")
handler.close()
```

### Option 4: Direct Configuration (No File)
```python
from CM_DB_Handler import CMDatabaseHandler

handler = CMDatabaseHandler(
    server="ROBINDWH.ROBINHQ.COM",
    database="RHQ_Andalusia_Group",
    username="your_username",
    password="your_password"
)

handler.connect()
df = handler.execute_query_from_file("SQL/CM_users.sql")
handler.save_to_excel(df, "output.xlsx")
handler.close()
```

## Usage

### Quick Start (Standalone Script)
```bash
python load_and_export.py
```

This will:
- Load credentials from `passcode.json` (or use defaults)
- Connect to the database
- Execute the query from `SQL/CM_users.sql`
- Export results to `CM_Users_YYYYMMDD_HHMMSS.xlsx`
- Display a summary of exported data

### Using from Command Line with Output to Specific File
```bash
# Edit load_and_export.py to customize output filename
python load_and_export.py
```

## Output

The Excel file will contain:
- **Sheet name**: "CM_Users"
- **Columns**: UniqueId, CreationDateTime, EmailAddress, LastUpdatedDateTime
- **Data**: All rows from the Accounts table query

## Troubleshooting

### Connection Issues
- **Error**: "Driver not found"
  - Install ODBC Driver 17 for SQL Server
  - Verify driver path in passcode.json
  
- **Error**: "Named instance not recognized"
  - Verify server name format in passcode.json
  - Use `hostname` or `hostname\instance`

- **Error**: "Login failed"
  - Check username and password in passcode.json
  - Verify user has permissions on the database

### File Not Found
- Ensure `SQL/CM_users.sql` exists in the same directory structure
- Check file paths are relative to script location
- Verify `passcode.json` exists if using that method

### Permission Issues
- Verify database user has SELECT permissions on the Accounts table
- Check network connectivity to ROBINDWH.ROBINHQ.COM
- Ensure ODBC driver is properly installed

## Security Notes

⚠️ **Important**: Never commit `passcode.json` with real credentials to version control.
- Add `passcode.json` to `.gitignore`
- Store credentials securely
- Use environment variables in production environments

## Notes

- Query file location: `SQL/CM_users.sql`
- Output files are automatically timestamped
- No index column is included in the Excel export
- Headers are included in the first row
- Supports multiple database credentials in `passcode.json` via `DB_NAMES` keys
