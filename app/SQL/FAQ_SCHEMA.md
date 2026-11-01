# FAQ Table Schema Documentation

## Database: Service_Hub
## Table: dbo.FAQ

This document describes the expected schema for the FAQ table used by the QA system.

## Required Columns

The following columns are required by the FAQ validation system:

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `ID` | INT or VARCHAR | Unique identifier for the FAQ record | 12345 |
| `CustomerName` | NVARCHAR | Name of the customer/patient | "أحمد محمد" |
| `mobile_phone` | VARCHAR | Customer's mobile phone number | "555123456" |
| `BU` | VARCHAR | Business Unit code | "HJH", "SNB" |
| `Date` | DATE or VARCHAR | Date of the call/inquiry | "2024-01-15" |
| `Inquiry` | NVARCHAR | Brief title/summary of the inquiry | "استفسار عن المواعيد" |
| `Response` | NVARCHAR(MAX) | Agent's response to the inquiry | "تم التواصل..." |
| `AgentName` | NVARCHAR | Name of the agent who handled the call | "Fatima Ali" |
| `AgentEmail` | VARCHAR | Email address of the agent | "fatima.ali@hospital.com" |
| `End Call Result` | NVARCHAR | Status of the call resolution | "In Progress", "Closed" |
| `CST Inquiry / Request Details` | NVARCHAR(MAX) | Detailed description of the customer inquiry | Full detailed description |

## Additional Columns

These columns are fetched but not required for validation:

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `Attachments` | NVARCHAR | File attachments related to the inquiry |
| `Created_at` | DATETIME | Record creation timestamp |
| `CRM_x0020_BU` | VARCHAR | CRM Business Unit reference |
| `Modified_BY` | NVARCHAR | Last user who modified the record |
| `FAQID` | VARCHAR | FAQ category identifier |
| `Speciality` | NVARCHAR | Medical specialty related to inquiry |
| `field5` | NVARCHAR | Custom field 5 |
| `field8` | NVARCHAR | Custom field 8 |
| `FQA` | NVARCHAR | FAQ category |
| `Injected to CRM` | BIT | Flag indicating if injected to CRM |
| `Modified_at` | DATETIME | Last modification timestamp |

## Query Pattern

The FAQ validation system queries records using:

```sql
WHERE 
    (
        REPLACE(REPLACE(REPLACE([mobile_phone], ' ', ''), '-', ''), '+', '') = ?
        OR [mobile_phone] = ?
    )
    AND [Date] = ?
ORDER BY [Modified_at] DESC
```

### Query Parameters:
1. Normalized phone number (digits only, no formatting)
2. Original phone number (as provided)
3. Call date (in the format stored in the Date column)

## Phone Number Format

The system normalizes phone numbers before querying:
- Strips country code prefixes (00966, 966)
- Removes leading zero from local format
- Results in 9-digit Saudi mobile number
- Example: "0555123456" → "555123456"

## Date Format

The Date column should store dates in a consistent format that matches the call_date field from the call transcript metadata.

## Inquiry Fields

### Inquiry (Title)
- Short, descriptive title of the inquiry
- Should summarize the customer's main question
- Typically 5-15 words

### CST Inquiry / Request Details (Full Description)
- Comprehensive description of what the customer inquired about
- Should capture all details from the conversation
- Can be multiple sentences or paragraphs
- Should match the semantic content of the transcript

## Business Unit Codes

Common BU equivalences (these are treated as the same):
- HJH ↔ AHJ (Al Habib Jarir Hospital)
- SNB ↔ AFW (Specialized Northern Border / Al Farwaniyah)
- CHT ↔ LCH ↔ ALW (Chat / Lateral Chat / Al Lubayd)
- MKR ↔ ADC ↔ JDC (Makka / Advanced Diagnostic Center / Jeddah Diagnostic Center)

## End Call Result Values

Valid values for End Call Result:
- **"In Progress"**: The inquiry is still being processed, awaiting response from responsible department
- **"Closed"**: The inquiry has been fully resolved and closed
- Other values may exist but these are the primary ones validated

## Index Recommendations

For optimal query performance, consider adding indexes on:
- `mobile_phone` (for phone number lookups)
- `Date` (for date filtering)
- `Modified_at` (for ordering results)
- Composite index on `(mobile_phone, Date)` for combined filtering

## Data Quality Requirements

1. **Phone numbers** should be consistently formatted
2. **Dates** should be in the same format as call metadata
3. **Agent emails** should be valid and consistent
4. **Agent names** should match the format used in call transcripts
5. **Inquiry and CST Inquiry** fields should be populated for all records
6. **Response** can be empty for "In Progress" inquiries

## Migration Notes

When migrating from CSV to this SQL table:
- Ensure all CSV columns map to corresponding SQL columns
- Handle null values appropriately
- Maintain data type compatibility
- Preserve historical records
- Test phone number normalization matches between CSV and SQL

## Access Requirements

The application connects using credentials defined in `Passcode.json`:
- Database: Service_Hub
- User: bi
- Password: [secured in Passcode.json]
- Driver: ODBC Driver 17 for SQL Server
- Server: 172.31.8.178

Ensure the user has SELECT permissions on the FAQ table.
