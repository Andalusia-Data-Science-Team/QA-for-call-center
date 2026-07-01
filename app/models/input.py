from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


VALID_DEPARTMENTS = {
    "Scheduling",
    "Onboarding",
    "Helpdesk",
    "Follow-Ups",
    "Records",
}

# Maps Arabic keywords found in the transcript to their business-unit code.
# Keys are checked as substrings (case-insensitive for ASCII, exact for Arabic).
# Add new entries here to support additional branches.
BUSINESS_UNIT_KEYWORD_MAP: dict[str, str] = {
    "مكرونة": "MKR",
    "المكرونة": "MKR",
    "سلطان": "LCH",
    "سلطان": "LCH",
    "فرع سلطان": "LCH",
    "جدة": "LIVE",
    "حي الجامعة": "LIVE",
    "سنابل": "SNB",
    "السنابل": "SNB",
    "صحة المرأة": "ALW",
    "المرأة": "ALW",
    "صحة الطفل": "AKW",
    "الطفل": "AKW",

}


class CallTranscript(BaseModel):
    call_id: str = Field(..., description="Unique identifier for the call.")
    agent_name: str = Field(..., description="Name of the virtual assistant who handled the call.")
    Patient_Phone: str = Field(..., description="Phone number of the patient.") 
    call_date: str = Field(..., description="Date of the call in YYYY-MM-DD format.")
    call_duration_seconds: int = Field(..., ge=0, description="Duration of the call in seconds.")
    department: str = Field(..., description="Department that handled the call.")
    transcript: str = Field(..., min_length=1, description="Full multi-turn call transcript.")
    business_unit: Optional[str] = Field(
        default=None,
        description="Business unit code auto-detected from the transcript (e.g. 'MKR', 'LCH'). "
                    "Set to None when no known branch keyword is found.",
    )

    @field_validator("call_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("call_date must be in YYYY-MM-DD format")
        return v

    @field_validator("transcript")
    @classmethod
    def transcript_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("transcript cannot be blank")
        return v.strip()
    
    @field_validator("Patient_Phone")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"\+?\d{9,15}", v):
            raise ValueError("Patient_Phone must be a valid phone number with 9 to 15 digits, optionally starting with +")
        v = v[4:] if v.startswith("+966") else v  # Remove +966 if present
        return v

    @model_validator(mode="after")
    def detect_business_unit_from_transcript(self) -> "CallTranscript":
        """
        Auto-detect the business unit by scanning the transcript for known
        Arabic branch keywords defined in BUSINESS_UNIT_KEYWORD_MAP.

        The first matching keyword wins (longest-match ordering is preserved
        by dict insertion order in Python 3.7+, so put more-specific phrases
        before shorter ones in BUSINESS_UNIT_KEYWORD_MAP).

        If no keyword is found and business_unit was not explicitly supplied,
        it remains None.
        """
        if self.business_unit is not None:
            # Caller explicitly set it — respect that value as-is.
            return self

        transcript_text = self.transcript or ""
        for keyword, code in BUSINESS_UNIT_KEYWORD_MAP.items():
            if keyword in transcript_text:
                self.business_unit = code
                break

        return self


class BatchCallTranscripts(BaseModel):
    calls: list[CallTranscript] = Field(
        ..., min_length=1, max_length=50, description="List of call transcripts to analyze."
    )
