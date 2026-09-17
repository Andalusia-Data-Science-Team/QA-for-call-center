# 📊 QA Logs & Analytics Dashboard

## Overview

The **Logs Dashboard** provides a comprehensive interface to search, filter, and analyze QA call evaluation logs. It aggregates data from multiple log files (`node_consumption.log`, `overall_consumption.log`, `node_output.log`) to provide unified insights into call quality metrics, LLM usage, costs, and agent performance.

---

## Features

### 🔍 **Search & Filter Capabilities**

- **Call ID**: Search by exact or partial call ID
- **Agent Email**: Filter by agent email address
- **Mobile Number**: Search for calls involving specific customer mobile numbers
- **Date Range**: Filter by call date (from/to)
- **Assessment Status**: Filter by overall assessment (pass, needs_review, escalate, error)

### 📈 **Real-time Statistics**

- Total calls analyzed
- Total LLM costs (USD)
- Total tokens consumed (prompt + completion)
- Average nodes executed per call
- Pass rate percentage
- Breakdown by assessment category

### 📋 **Detailed Call Insights**

For each call, view:
- **Node Execution Timeline**: Sequential execution of all graph nodes
- **LLM Usage**: Which nodes used LLM, models used, tokens consumed
- **Cost Breakdown**: Per-node and total costs
- **Evaluation Results**: Full QA assessment output with compliance flags
- **Agent Performance**: Agent name, classification, profiling comments
- **Timestamps**: When the call was analyzed

---

## Architecture

### Log Files Structure

```
logs/
├── node_consumption.log      # Per-node LLM consumption metrics
├── overall_consumption.log   # Aggregated call-level metrics
└── node_output.log           # Node execution outputs and evaluation data
```

### Components

1. **`app/services/log_parser.py`**
   - Core parsing logic for JSONL log files
   - Aggregates logs by call_id
   - Provides search and filtering methods
   - Calculates statistics

2. **`app/templates/logs-dashboard.html`**
   - Interactive web interface
   - Real-time search and filtering
   - Expandable detail views
   - Responsive design

3. **`app/main.py` (API Endpoints)**
   - `/logs-dashboard` - Dashboard UI
   - `/api/logs/search` - Search logs with filters
   - `/api/logs/call/{call_id}` - Get specific call details
   - `/api/logs/recent` - Get recent calls
   - `/api/logs/agent-summary` - Agent-specific statistics
   - `/api/logs/date-range-summary` - Date range analytics

---

## API Endpoints

### 1. Search Logs

**GET** `/api/logs/search`

**Query Parameters:**
- `call_id` (optional): Filter by call ID (partial match)
- `agent_email` (optional): Filter by agent email
- `mobile` (optional): Filter by mobile number
- `date_from` (optional): Start date in `YYYY-MM-DD` format
- `date_to` (optional): End date in `YYYY-MM-DD` format
- `assessment` (optional): Filter by status (pass, needs_review, escalate, error)

**Response:**
```json
{
  "results": [
    {
      "call_id": "C1C79E32-C4B1-F111-A6A7-000D3AA9D522",
      "agent_name": "Aya Mansour",
      "overall_assessment": "needs_review",
      "cost_usd": 0.0413895,
      "total_tokens": 30334,
      "node_count": 45,
      "timestamp": "2026-09-17T08:51:26.871272+00:00",
      "node_consumption": [...],
      "consumption": {...},
      "evaluation": {...}
    }
  ],
  "statistics": {
    "total_calls": 1,
    "total_cost_usd": 0.0413895,
    "total_tokens": 30334,
    "avg_nodes_per_call": 45.0,
    "pass_count": 0,
    "needs_review_count": 1,
    "escalate_count": 0,
    "error_count": 0,
    "pass_rate": "0.0%"
  },
  "total_results": 1
}
```

### 2. Get Call Detail

**GET** `/api/logs/call/{call_id}`

**Response:** Detailed call information including all node executions, consumption metrics, and evaluation output.

### 3. Get Recent Calls

**GET** `/api/logs/recent?limit=50`

**Response:** List of most recent calls (sorted by timestamp).

### 4. Agent Summary

**GET** `/api/logs/agent-summary?agent_email=agent@example.com`

**Response:** Statistics and recent calls for a specific agent.

### 5. Date Range Summary

**GET** `/api/logs/date-range-summary?date_from=2026-09-01&date_to=2026-09-17`

**Response:** Aggregated statistics for the specified date range.

---

## Usage Examples

### Example 1: Search by Call ID

```bash
curl "http://localhost:8005/api/logs/search?call_id=C1C79E32"
```

### Example 2: Filter by Agent and Date

```bash
curl "http://localhost:8005/api/logs/search?agent_email=aya.mansour&date_from=2026-09-16&date_to=2026-09-17"
```

### Example 3: Get Escalated Calls

```bash
curl "http://localhost:8005/api/logs/search?assessment=escalate"
```

### Example 4: Search by Mobile Number

```bash
curl "http://localhost:8005/api/logs/search?mobile=566617861"
```

---

## Web Interface Usage

### Accessing the Dashboard

1. Navigate to: `http://localhost:8005/logs-dashboard`
2. Or use the "📊 Logs Dashboard" button from:
   - QA Supervisor page
   - Agents Dashboard page

### Using the Filters

1. **Enter search criteria** in the filter fields:
   - Leave fields empty to search all
   - Use partial matches for Call ID and Agent Email
   - Use exact matches for Mobile Number

2. **Click "🔍 Search Logs"** to execute the search

3. **View Results**:
   - See summary statistics at the top
   - Browse the results table
   - Click "View Details" to expand call information

4. **Clear Filters**: Click "Clear Filters" to reset

### Interpreting Results

- **Green badge** (Pass): Call met all quality standards
- **Yellow badge** (Needs Review): Minor issues or data quality concerns
- **Red badge** (Escalate): Critical violations requiring immediate attention
- **Blue badge** (Error): System error during analysis

---

## Log Format Details

### Node Consumption Log

Each line represents one node execution:
```json
{
  "timestamp_utc": "2026-09-17T08:51:26.871272+00:00",
  "call_id": "C1C79E32-C4B1-F111-A6A7-000D3AA9D522",
  "node": "validate_crm_lead",
  "sequence": 38,
  "execution_count": 1,
  "provider": "openrouter",
  "model": "google/gemini-3.7-flash",
  "request_count": 1,
  "prompt_tokens": 2351,
  "completion_tokens": 1157,
  "total_tokens": 3508,
  "cost_usd": 0.006102,
  "uses_llm": true
}
```

### Overall Consumption Log

One line per completed call:
```json
{
  "timestamp_utc": "2026-09-17T08:51:26.871272+00:00",
  "call_id": "C1C79E32-C4B1-F111-A6A7-000D3AA9D522",
  "provider": "openrouter",
  "model": "google/gemini-3.7-flash",
  "request_count": 8,
  "prompt_tokens": 24121,
  "completion_tokens": 6213,
  "total_tokens": 30334,
  "cost_usd": 0.0413895,
  "node_count": 45,
  "llm_node_count": 8
}
```

### Node Output Log

Contains execution status and output for each node, including the final evaluation result.

---

## Performance Considerations

### Log File Size

- Logs are append-only and grow continuously
- Consider implementing log rotation for production use
- Large log files (>100MB) may slow down search operations

### Optimization Tips

1. **Use specific filters** to narrow search scope
2. **Filter by date range** when searching large time periods
3. **Use call_id** when looking for specific calls (fastest)
4. **Limit recent calls** queries with the `limit` parameter

### Future Enhancements

- [ ] Log rotation and archival
- [ ] Database indexing for faster searches
- [ ] Real-time log streaming
- [ ] Export to Excel/CSV
- [ ] Advanced analytics (trends, charts)
- [ ] Comparison views (agent vs agent, date vs date)

---

## Troubleshooting

### "No results found"

**Possible causes:**
- Log files are empty or missing
- Filters are too restrictive
- Date format is incorrect (use YYYY-MM-DD)

**Solutions:**
- Check that log files exist in `logs/` directory
- Try broadening your search criteria
- Verify date format

### "Log search failed"

**Possible causes:**
- Log files are corrupted
- Invalid JSON in log files
- Permission issues

**Solutions:**
- Check log file integrity
- Review application logs for details
- Verify file permissions

### Slow search performance

**Possible causes:**
- Large log files
- No filters applied (searching all records)

**Solutions:**
- Apply date range filters
- Use specific call_id or agent_email filters
- Consider log rotation/archival

---

## Integration with Existing Dashboards

The Logs Dashboard is integrated into the existing QA system:

1. **QA Supervisor**: Access via "📊 Logs Dashboard" button
2. **Agents Dashboard**: Access via navigation menu
3. **Direct URL**: `http://localhost:8005/logs-dashboard`

---

## Security Considerations

- Dashboard requires authentication (same as other QA pages)
- No data modification operations (read-only)
- Logs may contain sensitive call data - ensure proper access controls
- Consider implementing role-based access in production

---

## Maintenance

### Regular Tasks

1. **Monitor log file sizes**
2. **Implement log rotation** (e.g., daily or weekly)
3. **Archive old logs** to separate storage
4. **Backup log files** regularly
5. **Review search performance** as logs grow

### Log Rotation Example

```bash
# Rotate logs daily (add to cron)
cd /path/to/QA_System-main/logs
mv node_consumption.log node_consumption_$(date +%Y%m%d).log
mv overall_consumption.log overall_consumption_$(date +%Y%m%d).log
mv node_output.log node_output_$(date +%Y%m%d).log
touch node_consumption.log overall_consumption.log node_output.log
```

---

## Support

For issues or questions:
1. Check application logs: `logs/` directory
2. Review API error responses
3. Verify log file integrity
4. Contact system administrator

---

**Version**: 1.0.0  
**Last Updated**: 2026-09-17  
**Component**: QA Analysis System
