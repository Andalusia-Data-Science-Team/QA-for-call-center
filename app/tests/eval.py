#!/usr/bin/env python3
"""
Evaluation script — runs all sample transcripts against the live API
and validates that the system behaves as expected.

Usage:
    python tests/eval.py [--url http://localhost:8000]

Expected outcomes:
  - CALL-001-CLEAN       → overall_assessment: "pass",    escalation_required: false
  - CALL-002-ISSUES      → overall_assessment: "escalate", escalation_required: true
  - CALL-003-EDGE        → overall_assessment: "pass" or "needs_review", escalation_required: false
  - CALL-004-RUDE        → overall_assessment: "escalate", escalation_required: true
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

# ── Expectations ──────────────────────────────────────────────────────────────

EXPECTATIONS = {
    "CALL-001-CLEAN": {
        "overall_assessment": ["pass"],
        "escalation_required": False,
        "description": "Clean scheduling call — should pass with no escalation",
    },
    "CALL-002-ISSUES": {
        "overall_assessment": ["escalate"],
        "escalation_required": True,
        "description": "HIPAA violation: PHI disclosed without identity verification",
    },
    "CALL-003-EDGE": {
        "overall_assessment": ["pass", "needs_review"],
        "escalation_required": False,
        "description": "Disconnected call — cannot penalize agent, should not escalate",
    },
    "CALL-004-RUDE": {
        "overall_assessment": ["escalate", "needs_review"],
        "escalation_required": True,
        "description": "Rudeness + dangerous misinformation (false auth approval)",
    },
}

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check(condition: bool, label: str) -> bool:
    icon = f"{GREEN}✓{RESET}" if condition else f"{RED}✗{RESET}"
    print(f"    {icon}  {label}")
    return condition


def run_eval(base_url: str, transcript_dir: Path) -> int:
    """Returns number of failed assertions."""
    client = httpx.Client(base_url=base_url, timeout=120.0)
    failures = 0

    transcript_files = sorted(transcript_dir.glob("*.json"))
    if not transcript_files:
        print(f"{RED}No transcript files found in {transcript_dir}{RESET}")
        return 1

    print(f"\n{BOLD}Running evaluation against {base_url}{RESET}")
    print(f"Found {len(transcript_files)} transcript(s)\n")

    for path in transcript_files:
        with open(path) as f:
            payload = json.load(f)

        call_id = payload.get("call_id", path.stem)
        expectation = EXPECTATIONS.get(call_id)

        print(f"{BOLD}[{call_id}]{RESET} {path.name}")
        if expectation:
            print(f"  Scenario: {expectation['description']}")

        t0 = time.perf_counter()
        try:
            resp = client.post("/analyze-call", json=payload)
            elapsed = time.perf_counter() - t0
        except httpx.ConnectError:
            print(f"  {RED}CONNECTION ERROR — is the server running at {base_url}?{RESET}\n")
            failures += 1
            continue

        # ── Basic response checks ─────────────────────────────────────────
        ok = check(resp.status_code == 200, f"HTTP 200 (got {resp.status_code})")
        if not ok:
            print(f"  Response: {resp.text[:300]}\n")
            failures += 1
            continue

        try:
            data = resp.json()
        except json.JSONDecodeError:
            check(False, "Response is valid JSON")
            failures += 1
            continue

        check(True, f"Response is valid JSON ({elapsed:.2f}s)")

        # ── Schema checks ─────────────────────────────────────────────────
        required_keys = [
            "overall_assessment", "assessment_reasoning", "compliance_flags",
            "agent_performance", "escalation_required", "escalation_reason",
        ]
        for key in required_keys:
            if not check(key in data, f"Field '{key}' present"):
                failures += 1

        perf = data.get("agent_performance", {})
        for score in ["professionalism_score", "accuracy_score", "resolution_score"]:
            val = perf.get(score)
            if val is not None:
                in_range = 0.0 <= val <= 1.0
                check(in_range, f"agent_performance.{score} in [0,1]: {val}")
                if not in_range:
                    failures += 1

        # ── Expectation checks ────────────────────────────────────────────
        if expectation:
            assessment = data.get("overall_assessment")
            allowed = expectation["overall_assessment"]
            ok = check(assessment in allowed, f"overall_assessment '{assessment}' in {allowed}")
            if not ok:
                failures += 1

            escalate = data.get("escalation_required")
            ok = check(
                escalate == expectation["escalation_required"],
                f"escalation_required == {expectation['escalation_required']} (got {escalate})",
            )
            if not ok:
                failures += 1

            # If escalation expected, reason should not be null
            if expectation["escalation_required"]:
                reason = data.get("escalation_reason")
                ok = check(
                    reason is not None and len(reason) > 0,
                    f"escalation_reason is not null",
                )
                if not ok:
                    failures += 1

        # ── Summary output ────────────────────────────────────────────────
        print(f"\n  {BOLD}Result:{RESET} {data.get('overall_assessment')} | "
              f"escalate={data.get('escalation_required')}")
        print(f"  Reasoning: {data.get('assessment_reasoning', '')[:120]}...")
        flags = data.get("compliance_flags", [])
        if flags:
            print(f"  Flags ({len(flags)}):")
            for flag in flags:
                sev_color = RED if flag.get("severity") == "critical" else YELLOW
                print(f"    {sev_color}[{flag.get('severity')}]{RESET} "
                      f"{flag.get('type')}: {flag.get('description', '')[:80]}")
        print()

    # ── Final report ──────────────────────────────────────────────────────
    print("-" * 60)
    if failures == 0:
        print(f"{GREEN}{BOLD}All checks passed!{RESET}")
    else:
        print(f"{RED}{BOLD}{failures} check(s) failed.{RESET}")

    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate call QA API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument(
        "--transcripts",
        default=str(Path(__file__).parent.parent / "sample_transcripts"),
        help="Directory containing transcript JSON files",
    )
    args = parser.parse_args()

    failures = run_eval(args.url, Path(args.transcripts))
    sys.exit(0 if failures == 0 else 1)
