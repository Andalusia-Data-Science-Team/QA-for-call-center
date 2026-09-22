"""
Log parser service for aggregating and searching QA analysis logs.

Parses node_consumption.log, overall_consumption.log, and node_output.log
to provide unified search and analytics capabilities.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class LogParser:
    """Parse and aggregate QA analysis logs."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.node_consumption_file = self.logs_dir / "node_consumption.log"
        self.overall_consumption_file = self.logs_dir / "overall_consumption.log"
        self.node_output_file = self.logs_dir / "node_output.log"

    def parse_all_logs(self) -> Dict[str, Any]:
        """Return the latest successful evaluation for every evaluated chat.

        ``aggregate_results`` is the authoritative record that an evaluation
        completed. Consumption logs are supplementary: they must never hide a
        completed evaluation when their records are absent or incomplete.
        """
        calls: Dict[str, Dict[str, Any]] = {}
        node_outputs = self._parse_jsonl(self.node_output_file)

        # A successful aggregate record means the chat was evaluated, even if
        # no consumption data was emitted for that run.
        for entry in node_outputs:
            if entry.get("node") != "aggregate_results" or entry.get("status") != "success":
                continue
            call_id = entry.get("call_id")
            evaluation_date = entry.get("timestamp_utc") or ""
            if not call_id or evaluation_date < (calls.get(call_id, {}).get("evaluation_date") or ""):
                continue

            result = (entry.get("output") or {}).get("result") or {}
            calls[call_id] = {
                "call_id": call_id,
                "node_consumption": [],
                "consumption": {},
                "evaluation": result,
                "evaluation_date": entry.get("timestamp_utc"),
                # Retained for clients that previously used this field.
                "timestamp": entry.get("timestamp_utc"),
                "overall_assessment": result.get("overall_assessment"),
                "agent_name": result.get("agent_name"),
                "compliance_flags": result.get("compliance_flags", []),
            }

        # Consumption enriches evaluation details but never controls whether a
        # completed evaluation is visible. Keep its newest run per chat.
        latest_consumption_timestamp: Dict[str, str] = {}
        for entry in self._parse_jsonl(self.node_consumption_file):
            call_id = entry.get("call_id")
            if call_id not in calls:
                continue
            timestamp = entry.get("timestamp_utc") or ""
            latest = latest_consumption_timestamp.get(call_id, "")
            if timestamp > latest:
                latest_consumption_timestamp[call_id] = timestamp
                calls[call_id]["node_consumption"] = [entry]
            elif timestamp == latest:
                calls[call_id]["node_consumption"].append(entry)

        latest_overall_timestamp: Dict[str, str] = {}
        for entry in self._parse_jsonl(self.overall_consumption_file):
            call_id = entry.get("call_id")
            if call_id not in calls:
                continue
            timestamp = entry.get("timestamp_utc") or ""
            if timestamp < latest_overall_timestamp.get(call_id, ""):
                continue
            latest_overall_timestamp[call_id] = timestamp
            calls[call_id]["consumption"] = entry
            calls[call_id]["cost_usd"] = entry.get("cost_usd") or 0
            calls[call_id]["total_tokens"] = entry.get("total_tokens") or 0
            calls[call_id]["node_count"] = entry.get("node_count") or 0

        for entry in node_outputs:
            call_id = entry.get("call_id")
            if entry.get("node") == "finalize" and call_id in calls:
                timestamp = entry.get("timestamp_utc") or ""
                if timestamp >= (calls[call_id].get("completed_at") or ""):
                    calls[call_id]["completed_at"] = entry.get("timestamp_utc")

        return calls

    def search_logs(
        self,
        call_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_email: Optional[str] = None,
        mobile: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        assessment: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Search logs with filters and return results with statistics.
        
        Args:
            call_id: Filter by call ID (partial match)
            agent_name: Filter by agent name (partial match)
            agent_email: Filter by agent email (partial match)
            mobile: Filter by mobile number (searches in evaluation data)
            date_from: Filter by start date (YYYY-MM-DD)
            date_to: Filter by end date (YYYY-MM-DD)
            assessment: Filter by overall_assessment (pass, needs_review, escalate, error)
        
        Returns:
            Dict with 'results' and 'statistics' keys
        """
        all_calls = self.parse_all_logs()
        filtered = []

        for call_data in all_calls.values():
            # Filter by call_id
            if call_id and call_id.lower() not in call_data.get("call_id", "").lower():
                continue

            # Filter by agent_name
            agent_name_value = call_data.get("agent_name", "")
            
            if agent_name:
                if agent_name.lower() not in agent_name_value.lower():
                    continue
            
            # Filter by agent_email
            evaluation = call_data.get("evaluation", {})
            agent_email_in_eval = evaluation.get("agent_email_address", "")
            
            if agent_email:
                email_lower = agent_email.lower()
                if (email_lower not in agent_name_value.lower() and 
                    email_lower not in agent_email_in_eval.lower()):
                    continue

            # Filter by mobile number (search in evaluation data)
            if mobile:
                eval_str = json.dumps(evaluation).lower()
                if mobile not in eval_str:
                    continue

            # Filter by date range
            timestamp = call_data.get("evaluation_date") or call_data.get("timestamp")
            if timestamp:
                try:
                    # Parse ISO timestamp
                    call_date = datetime.fromisoformat(timestamp.replace("+00:00", ""))
                    
                    if date_from:
                        start_date = datetime.strptime(date_from, "%Y-%m-%d")
                        if call_date.date() < start_date.date():
                            continue
                    
                    if date_to:
                        end_date = datetime.strptime(date_to, "%Y-%m-%d")
                        if call_date.date() > end_date.date():
                            continue
                except (ValueError, AttributeError) as e:
                    logger.warning("Failed to parse timestamp %s: %s", timestamp, e)

            # Filter by assessment
            if assessment:
                if call_data.get("overall_assessment") != assessment:
                    continue

            filtered.append(call_data)

        # Sort by the actual evaluation time, newest first.
        filtered.sort(
            key=lambda x: x.get("evaluation_date") or x.get("timestamp") or "",
            reverse=True
        )

        # Calculate statistics
        statistics = self._calculate_statistics(filtered)

        total_results = len(filtered)
        if page is not None and page_size is not None:
            page = max(page, 1)
            page_size = max(page_size, 1)
            total_pages = max((total_results + page_size - 1) // page_size, 1)
            page = min(page, total_pages)
            start = (page - 1) * page_size
            results = filtered[start:start + page_size]
            pagination = {
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_results": total_results,
            }
        else:
            results = filtered
            pagination = None

        response = {
            "results": results,
            "statistics": statistics,
            "total_results": total_results,
        }
        if pagination:
            response["pagination"] = pagination
        return response

    def get_call_detail(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific call."""
        all_calls = self.parse_all_logs()
        return all_calls.get(call_id)

    def _parse_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a JSONL (JSON Lines) file."""
        if not file_path.exists():
            logger.warning("Log file not found: %s", file_path)
            return []

        entries = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        logger.error(
                            "Failed to parse line %d in %s: %s",
                            line_num, file_path.name, e
                        )
        except Exception as e:
            logger.error("Failed to read %s: %s", file_path, e)

        return entries

    def _calculate_statistics(self, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics from filtered calls."""
        if not calls:
            return {
                "total_calls": 0,
                "total_cost_usd": 0,
                "total_tokens": 0,
                "avg_nodes_per_call": 0,
                "pass_count": 0,
                "needs_review_count": 0,
                "escalate_count": 0,
                "error_count": 0,
                "pass_rate": "0%",
            }

        # JSON logs can contain explicit null values for incomplete or
        # non-LLM runs. dict.get(..., 0) does not replace an existing None,
        # so normalize falsy numeric values before summing.
        total_cost = sum(c.get("cost_usd") or 0 for c in calls)
        total_tokens = sum(c.get("total_tokens") or 0 for c in calls)
        total_nodes = sum(c.get("node_count") or 0 for c in calls)
        
        assessment_counts = defaultdict(int)
        for call in calls:
            assessment = call.get("overall_assessment", "unknown")
            assessment_counts[assessment] += 1

        pass_count = assessment_counts.get("pass", 0)
        pass_rate = (pass_count / len(calls) * 100) if calls else 0

        return {
            "total_calls": len(calls),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "avg_nodes_per_call": round(total_nodes / len(calls), 2) if calls else 0,
            "pass_count": pass_count,
            "needs_review_count": assessment_counts.get("needs_review", 0),
            "escalate_count": assessment_counts.get("escalate", 0),
            "error_count": assessment_counts.get("error", 0),
            "pass_rate": f"{pass_rate:.1f}%",
        }

    def get_recent_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recently evaluated calls."""
        all_calls = self.parse_all_logs()
        calls_list = list(all_calls.values())
        
        # Sort by timestamp (most recent first)
        calls_list.sort(
            key=lambda x: x.get("evaluation_date") or x.get("timestamp") or "",
            reverse=True
        )
        
        return calls_list[:limit]

    def get_agent_summary(self, agent_email: str) -> Dict[str, Any]:
        """Get summary statistics for a specific agent."""
        search_result = self.search_logs(agent_email=agent_email)
        return {
            "agent_email": agent_email,
            "statistics": search_result["statistics"],
            "recent_calls": search_result["results"][:10],
        }

    def get_date_range_summary(
        self, date_from: str, date_to: str
    ) -> Dict[str, Any]:
        """Get summary for a date range."""
        search_result = self.search_logs(date_from=date_from, date_to=date_to)
        
        # Group by assessment
        by_assessment = defaultdict(list)
        for call in search_result["results"]:
            assessment = call.get("overall_assessment", "unknown")
            by_assessment[assessment].append(call)
        
        return {
            "date_from": date_from,
            "date_to": date_to,
            "statistics": search_result["statistics"],
            "by_assessment": {
                k: len(v) for k, v in by_assessment.items()
            },
        }
