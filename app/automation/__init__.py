"""
Automated QA sampling and evaluation.

See docs/AUTOMATION_IMPLEMENTATION_PLAN.md. Nothing in this package runs
unless QA_AUTOMATION_ENABLED is true, and importing it reads no configuration
and opens no connections — call `automation_settings()` to load settings.
"""

from app.automation.config import AutomationSettings, automation_settings

__all__ = ["AutomationSettings", "automation_settings"]
