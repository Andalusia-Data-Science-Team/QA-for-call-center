# FAQ Validation Migration: CSV to SQL Database

## Overview
This document describes the migration of the FAQ validation system from CSV file-based storage to SQL Server database queries.

## Changes Made

### 1. SQL Query File (`app/SQL/faq_query.sql`)
- **New File**: Created SQL query to fetch FAQ records from `Service_Hub.dbo.FAQ` table
- **Key Features**:
  - Fetches all required columns including `Inquiry` and `CST Inquiry / Request Details`
  - Filters by normalized phone number (handles different formats)
  - Filters by exact call date
  - Orders by `Modified_at DESC` to get the most recent record first
  - Limits to top 25,000 records for performance

### 2. Database Configuration (`app/Passcode.json`)
- **Added**: `SERVICE_HUB` database connection configuration
  - Server: 172.31.8.178
  - Database: Service_Hub
  - Uses existing BI credentials
  - ODBC Driver 17 for SQL Server

### 3. FAQ Validation Module (`app/FAQs/faq_validation.py`)

#### New Imports
- Added `json` for reading configuration
- Added `pyodbc` for database connectivity

#### Updated Constants
- Replaced `FAQ_CSV_PATH` with `FAQ_SQL_PATH`
- Added `FAQ_PASSCODE_PATH` for database credentials
- Updated `FAQ_REQUIRED_COLUMNS` to include `"CST Inquiry / Request Details"`

#### New Functions
- `_load_db_credentials(db_key)`: Loads database credentials from Passcode.json
- `_get_db_connection(db_key)`: Creates pyodbc connection to SQL Server

#### Updated Functions

**`lookup_faq_record()`**:
- Changed from reading CSV file to querying SQL database
- Parameters changed: `csv_path` → `db_key` (default: "SERVICE_HUB")
- Uses pyodbc to execute parameterized SQL query
- Converts SQL result rows to dictionaries
- Maintains same matching logic (phone, date, agent identity)
- Returns same response structure for backward compatibility

**`detect_faq_escalation()`**:
- Enhanced to ONLY check agent messages for escalation keywords
- Uses `_agent_message_text()` to extract agent-only messages
- Ensures that only agent claims of escalation are detected (not patient questions)

**`_agent_message_text()`**:
- Simplified to return only labeled agent messages
- Removed fallback to full transcript to ensure accuracy

### 4. Agent Nodes (`app/agent/nodes.py`)
- Updated `validate_faq_record()` function signature:
  - Changed parameter: `csv_path: Path | None = None` → `db_key: str = "SERVICE_HUB"`
  - Updated function call to `lookup_faq_record()` with new parameter
- Updated error message: "FAQ CSV unavailable" → "FAQ database unavailable"

### 5. Validation Prompt (`app/prompts/qa_prompt.py`)

**`build_faq_validation_prompt()`**:
- Enhanced validation rule #4 to explicitly check both:
  - `Inquiry` field (FAQ title/summary)
  - `CST Inquiry / Request Details` field (full inquiry description)
- Added clear guidance on semantic comparison vs exact wording
- Updated rule #9 to include `"CST Inquiry / Request Details"` in field checks

### 6. Test Script (`test_faq_db.py`)
- **New File**: Created test script to verify:
  - Database connectivity
  - FAQ record lookup
  - Escalation detection with different transcript formats
  - Agent vs Patient message filtering

## Key Features

### Phone Number Normalization
The SQL query handles multiple phone number formats:
- Removes spaces, dashes, and plus signs
- Matches both normalized and original formats
- Example: "555 555 555", "+966555555555", "0555555555" all match

### Inquiry Field Validation
The system now validates TWO inquiry fields:
1. **Inquiry**: Brief title/summary of the customer's request
2. **CST Inquiry / Request Details**: Detailed description of the inquiry

Both fields are:
- Included in the fetched FAQ record
- Passed to the LLM for validation
- Checked semantically against the transcript

### Agent Message Filtering
Enhanced escalation detection to ensure accuracy:
- Only checks messages explicitly labeled as from the agent
- Ignores patient questions about escalation
- Prevents false positives from patient inquiries

## Migration Benefits

1. **Real-time Data**: No need to export and update CSV files
2. **Scalability**: Database can handle millions of records efficiently
3. **Consistency**: Single source of truth for FAQ records
4. **Performance**: Indexed queries are faster than CSV parsing
5. **Integration**: Easier to integrate with other database systems
6. **Security**: Database credentials managed securely
7. **Audit Trail**: Database provides modification timestamps

## Testing

Run the test script to verify the migration:
```bash
cd /home/ai/Workspace/Rafik/QA_System-main
python test_faq_db.py
```

Expected output:
- Escalation detection tests with different transcript formats
- Database connection and query execution
- FAQ record retrieval with all fields

## Backward Compatibility

The migration maintains backward compatibility:
- Same return structure from `lookup_faq_record()`
- Same status codes: "found", "not_found", "unavailable"
- Same error handling patterns
- Existing test cases should still pass with minimal updates

## Database Requirements

Ensure the SQL Server has:
- `Service_Hub` database accessible
- `dbo.FAQ` table with all required columns
- Proper permissions for the BI user
- Network connectivity from the application server

## Next Steps

1. Test the migration in a development environment
2. Verify all FAQ records are accessible via SQL query
3. Update any remaining references to CSV in documentation
4. Monitor query performance and add indexes if needed
5. Consider adding connection pooling for better performance
6. Update unit tests to work with SQL instead of CSV fixtures
