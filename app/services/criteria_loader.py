"""
CriteriaLoader — loads YAML policy files and renders them as compact,
token-efficient plain-text blocks for injection into LLM prompts.

TOKEN OPTIMISATION STRATEGY
────────────────────────────
Raw YAML is never sent to the LLM.  Only the semantically meaningful content
is extracted and formatted as tight structured text.  Fields that add no
analytical value are dropped:
  ✗  version, last_updated, effective_date, source, category, note
  ✗  YAML keys / indentation / hyphens
  ✗  "Other" catch-all violation items (implied, not useful)
  ✗  Repeated severity labels already conveyed by the pillar type header

Typical savings vs raw YAML: 40–55 % fewer tokens per file.

Usage
─────
    loader = CriteriaLoader()

    # Returns a compact string ready for prompt injection
    behavioral  = loader.behavioral(department="Neurology")
    pillars     = loader.compliance_pillars()
    scripts     = loader.script_templates()
    weights     = loader.scoring_weights()
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import arabic_reshaper
from bidi.algorithm import get_display
import yaml

logger = logging.getLogger(__name__)


def _fix_arabic(text: str) -> str:
    """Reshape and apply bidi algorithm to fix reversed Arabic rendering."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

_CRITERIA_ROOT = Path(__file__).parent.parent / "criteria"


# ─────────────────────────────────────────────────────────────────────────────
# Internal YAML helpers
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def _load_yaml(rel_path: str) -> dict[str, Any]:
    """Load and cache a YAML file relative to the criteria root."""
    full = _CRITERIA_ROOT / rel_path
    if not full.exists():
        logger.warning("CriteriaLoader: file not found — %s", full)
        return {}
    with full.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _list_bullets(items: list[Any], prefix: str = "  • ") -> str:
    return "\n".join(f"{prefix}{i}" for i in items if i != "Other")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class CriteriaLoader:
    """
    Single entry-point for all criteria / policy content.
    All public methods return a compact plain-text string.
    """

    # ── 1. Behavioral criteria ────────────────────────────────────────────

    def behavioral(self, department: str = "general") -> str:
        """
        Return a compact behavioral-standards block for the given department.
        Falls back to 'general' if the department file doesn't exist.
        """
        dept_key = department.lower().replace(" ", "_")
        data = _load_yaml(f"policies/behavioral/{dept_key}.yaml")
        if not data:
            data = _load_yaml("policies/behavioral/general.yaml")
            dept_key = "general"

        lines: list[str] = [f"# BEHAVIORAL STANDARDS — {dept_key.upper()}"]

        # Script compliance
        sc = data.get("script_compliance", {})
        if sc:
            lines.append("\n## Required Script Compliance")
            for section, cfg in sc.items():
                lines.append(f"### {section.capitalize()}")
                if must := cfg.get("must_include"):
                    lines.append("  Required phrases: " + " | ".join(f'"{p}"' for p in must))
                if prohibited := cfg.get("prohibited"):
                    lines.append("  Prohibited phrases: " + " | ".join(f'"{p}"' for p in prohibited))
                if preferred := cfg.get("preferred_phrases"):
                    lines.append("  Preferred: " + " | ".join(f'"{p}"' for p in preferred))

        # Professionalism
        prof = data.get("professionalism", {})
        if prof:
            lines.append("\n## Professionalism")
            tone = prof.get("tone", {})
            if req := tone.get("required_qualities"):
                lines.append("  Required tone: " + ", ".join(req))
            if banned := tone.get("prohibited_tone"):
                lines.append("  Prohibited tone: " + ", ".join(banned))
            lang = prof.get("language", {})
            if banned_phrases := lang.get("prohibited_phrases"):
                lines.append("  Prohibited phrases: " + " | ".join(f'"{p}"' for p in banned_phrases))

        # Empathy
        emp = data.get("empathy", {})
        if emp:
            lines.append("\n## Empathy & Active Listening")
            if must := emp.get("must_demonstrate"):
                lines.append(_list_bullets(must))
            al = emp.get("active_listening", {})
            if techniques := al.get("techniques"):
                lines.append("  Techniques: " + ", ".join(techniques))
            if al_banned := al.get("prohibited"):
                lines.append("  Avoid: " + ", ".join(al_banned))

        # Prohibited behaviors
        pb = data.get("prohibited_behaviors", {})
        if pb:
            lines.append("\n## Prohibited Behaviors")
            for severity_label, items in pb.items():
                lines.append(f"  [{severity_label.replace('_', ' ').title()}]")
                lines.append(_list_bullets(items, prefix="    – "))

        # Red flag phrases
        if rf := data.get("red_flag_phrases"):
            lines.append("\n## Red-Flag Phrases (immediate flag if observed)")
            lines.append("  " + " | ".join(f'"{p}"' for p in rf))

        # Department-specific extras (neurology / pain management)
        for extra_key in ("clinical_standards", "neurology_specific", "pain_specific"):
            extra = data.get(extra_key, {})
            if extra:
                lines.append(f"\n## {extra_key.replace('_', ' ').title()}")
                lines.append(self._render_dict_compact(extra))

        return "\n".join(lines)

    # ── 2. Compliance pillars ─────────────────────────────────────────────

    def compliance_pillars(self) -> str:
        """
        Return a compact compliance-pillar checklist from regulations.yaml.
        Strips all ids, type repetitions, and 'Other' items.
        """
        data = _load_yaml("policies/regulations/compliance_regulations.yaml")
        pillars = data.get("pillars", {})
        if not pillars:
            return "# COMPLIANCE PILLARS\n(none loaded)"

        lines: list[str] = ["# COMPLIANCE PILLARS (Andalusia Hospitals)"]
        # Group by severity type for a compact layout
        groups: dict[str, list[tuple[str, list[str]]]] = {
            "C2Com": [], "C2C": [], "C2B": [], "NC": []
        }
        for pillar_key, pillar in pillars.items():
            ptype = pillar.get("severity", "NC")
            violations = [
                item["violation"]
                for item in pillar.get("items", [])
                if item.get("violation", "Other") != "Other"
            ]
            if violations:
                groups.setdefault(ptype, []).append((pillar.get("original_name", pillar_key), violations))

        labels = {
            "C2Com": "C2Com — Critical to Compliance",
            "C2C":   "C2C   — Critical to Client / End-User",
            "C2B":   "C2B   — Critical to Business",
            "NC":    "NC    — Non-Critical",
        }
        for ptype, label in labels.items():
            group = groups.get(ptype, [])
            if not group:
                continue
            lines.append(f"\n## {label}")
            for pillar_name, violations in group:
                lines.append(f"\n### {pillar_name}")
                lines.append(_list_bullets(violations))
        return "\n".join(lines)
    
    # ── 3. Reservation pillars ─────────────────────────────────────────────

    def reservation_pillars(self) -> str:
        """
        Return a compact reservation-pillar checklist from regulations.yaml.
        Strips all ids, type repetitions, and 'Other' items.
        """
        data = _load_yaml("policies/regulations/reservation_regulations.yaml")
        pillars = data.get("pillars", {})
        if not pillars:
            return "# RESERVATION PILLARS\n(none loaded)"

        lines: list[str] = ["# RESERVATION PILLARS (Andalusia Hospitals)"]
        # Group by severity type for a compact layout
        groups: dict[str, list[tuple[str, list[str]]]] = {
            "C2Com": [], "C2C": [], "C2B": [], "NC": []
        }
        for pillar_key, pillar in pillars.items():
            ptype = pillar.get("severity", "NC")
            violations = [
                item["violation"]
                for item in pillar.get("items", [])
                if item.get("violation", "Other") != "Other"
            ]
            if violations:
                groups.setdefault(ptype, []).append((pillar.get("original_name", pillar_key), violations))

        labels = {
            "C2Com": "C2Com — Critical to Compliance",
            "C2C":   "C2C   — Critical to Client / End-User",
            "C2B":   "C2B   — Critical to Business",
            "NC":    "NC    — Non-Critical",
        }
        for ptype, label in labels.items():
            group = groups.get(ptype, [])
            if not group:
                continue
            lines.append(f"\n## {label}")
            for pillar_name, violations in group:
                lines.append(f"\n### {pillar_name}")
                lines.append(_list_bullets(violations))
        return "\n".join(lines)

    # ── 4. Offer recommendation pillars ──────────────────────────────────

    def offer_pillars(self) -> str:
        """
        Return a compact offer-recommendation pillar checklist from
        offer_regulations.yaml.  Strips ids, type repetitions, and 'Other'
        items.  Also appends the evaluation-guidance section so the LLM knows
        exactly when to penalise and when not to.
        """
        data = _load_yaml("policies/regulations/offer_regulations.yaml")
        pillars = data.get("pillars", {})
        guidance = data.get("evaluation_guidance", {})
        if not pillars:
            return "# OFFER RECOMMENDATION PILLARS\n(none loaded)"

        lines: list[str] = ["# OFFER RECOMMENDATION PILLARS (Andalusia Hospitals)"]
        groups: dict[str, list[tuple[str, list[str]]]] = {
            "C2Com": [], "C2C": [], "C2B": [], "NC": []
        }
        for pillar_key, pillar in pillars.items():
            ptype = pillar.get("severity", "NC")
            violations = [
                item["violation"]
                for item in pillar.get("items", [])
                if item.get("violation", "Other") != "Other"
            ]
            if violations:
                groups.setdefault(ptype, []).append(
                    (pillar.get("original_name", pillar_key), violations)
                )

        labels = {
            "C2Com": "C2Com — Critical to Compliance",
            "C2C":   "C2C   — Critical to Client / End-User",
            "C2B":   "C2B   — Critical to Business",
            "NC":    "NC    — Non-Critical",
        }
        for ptype, label in labels.items():
            group = groups.get(ptype, [])
            if not group:
                continue
            lines.append(f"\n## {label}")
            for pillar_name, violations in group:
                lines.append(f"\n### {pillar_name}")
                lines.append(_list_bullets(violations))

        # Append evaluation guidance
        if guidance:
            lines.append("\n## WHEN TO CHECK FOR OFFERS")
            for item in guidance.get("when_to_check", []):
                lines.append(f"  • {item}")
            lines.append("\n## WHEN NOT TO PENALISE")
            for item in guidance.get("when_NOT_to_penalise", []):
                lines.append(f"  • {item}")
            lines.append("\n## SCORING OUTCOMES")
            for outcome, cfg in guidance.get("scoring_outcomes", {}).items():
                flag_type = cfg.get("flag_type") or "none"
                desc = cfg.get("description", "")
                lines.append(f"  {outcome}: flag_type={flag_type} — {desc}")

        return "\n".join(lines)

    # ── 5. Script templates ───────────────────────────────────────────────

    def script_templates(self) -> str:
        """
        Return DCC script compliance criteria from the unified DCC_scripts.yaml file.
        """
        dcc_scripts = _load_yaml("scripts/DCC_scripts.yaml")
        lines: list[str] = ["# SCRIPT TEMPLATES"]
        
        # Render DCC script compliance evaluation criteria
        script_compliance = dcc_scripts.get("script_compliance", {})
        eval_criteria = script_compliance.get("evaluation_criteria", [])
        if eval_criteria:
            lines.append("\n## DCC Script Compliance Evaluation Criteria")
            for item in eval_criteria:
                order = item.get("order", "")
                stage = item.get("stage", "")
                script = item.get("script", "")
                notes = item.get("notes", "")
                
                lines.append(f"\n  [{order}] {stage}")
                if script:
                    formatted_script = _fix_arabic(script)
                    lines.append(f"    Script: {formatted_script}")
                if notes:
                    formatted_notes = _fix_arabic(notes)
                    lines.append(f"    Notes: {formatted_notes}")
        
        return "\n".join(lines)

    # ── 5. Scoring weights ────────────────────────────────────────────────

    def scoring_weights(self) -> str:
        """
        Return a compact scoring-weight reference.
        """
        data = _load_yaml("policies/scoring/weights.yaml")
        ow = data.get("overall_weights", {})
        beh = ow.get("behavioral", {})
        if not beh:
            return "# SCORING WEIGHTS\n(none loaded)"

        lines: list[str] = [
            "# SCORING WEIGHTS",
            f"  Minimum passing score: {ow.get('minimum_passing_score', 85.0)}%",
            f"  Critical violation deduction: -{ow.get('critical_violation_deduction', 10.0)}pts",
            "\n  Dimension weights:",
        ]
        for dim, w in beh.items():
            lines.append(f"    • {dim.replace('_', ' ').title():<22} {int(w * 100)}%")
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _render_dict_compact(d: dict, indent: int = 2) -> str:
        pad = " " * indent
        out: list[str] = []
        for k, v in d.items():
            if isinstance(v, list):
                out.append(f"{pad}{k}:")
                for item in v:
                    out.append(f"{pad}  • {item}")
            elif isinstance(v, dict):
                out.append(f"{pad}{k}:")
                out.append(CriteriaLoader._render_dict_compact(v, indent + 2))
            else:
                out.append(f"{pad}{k}: {v}")
        return "\n".join(out)
