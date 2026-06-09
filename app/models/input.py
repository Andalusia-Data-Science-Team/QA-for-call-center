from pydantic import BaseModel, Field, field_validator


VALID_DEPARTMENTS = {
    "Scheduling",
    "Onboarding",
    "Helpdesk",
    "Follow-Ups",
    "Records",
}


class CallTranscript(BaseModel):
    call_id: str = Field(..., description="Unique identifier for the call.")
    agent_name: str = Field(..., description="Name of the virtual assistant who handled the call.")
    call_date: str = Field(..., description="Date of the call in YYYY-MM-DD format.")
    call_duration_seconds: int = Field(..., ge=0, description="Duration of the call in seconds.")
    department: str = Field(..., description="Department that handled the call.")
    transcript: str = Field(..., min_length=1, description="Full multi-turn call transcript.")

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


class BatchCallTranscripts(BaseModel):
    calls: list[CallTranscript] = Field(
        ..., min_length=1, max_length=50, description="List of call transcripts to analyze."
    )
