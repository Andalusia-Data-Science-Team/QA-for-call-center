# eligibility_service.py

import json
import logging
import pandas as pd
from pathlib import Path
from sqlalchemy import text

from eligibility.eligibility import (
    create_json_payload,
    send_json_to_api,
    extract_code,
    extract_note,
    extract_outcome,
    parse_row,
    change_date,
    map_row,
)

from eligibility.etl_utils import get_conn_engine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EligibilityService:

    def __init__(self, passcode_path="eligibility/passcode.json"):

        with open(passcode_path, "r") as file:
            data_dict = json.load(file)

        self.db_names = data_dict["DB_NAMES"]

    def fetch_visit_data(self, visit_id: int) -> pd.Series:
        """
        Fetch single visit data from DB
        """

        query_path = (
                Path(__file__).resolve().parent.parent
                / "SQL"
                / "eligibility_enhanced.sql"
            )
        query = query_path.read_text()

        engine = get_conn_engine(self.db_names["LIVE"], logger)

        try:
            with engine.connect() as conn:

                result = conn.execute(
                    text(query),
                    {"visit_id": visit_id}
                )

                row = result.fetchone()

                if not row:
                    raise ValueError(f"visit_id={visit_id} not found")

                df = pd.DataFrame([row], columns=result.keys())

                return df.iloc[0]

        finally:
            engine.dispose()

    def prepare_row(self, row: pd.Series) -> pd.Series:
        """
        Apply existing transformations
        """

        row = map_row(row)

        row["start_date"] = change_date(row.get("start_date"))
        row["end_date"] = change_date(row.get("end_date"))
        row["date_of_birth"] = change_date(row.get("date_of_birth"))

        return row

    def generate_payload(self, row: pd.Series) -> dict:
        """
        Generate API payload
        """

        return create_json_payload(
            row=row,
            source="LIVE"
        )

    def call_eligibility_api(self, payload: dict) -> dict:
        """
        Send request to eligibility API
        """

        return send_json_to_api(payload)

    def parse_response(self, response: dict) -> dict:
        """
        Extract eligibility information
        """

        eligibility_class = extract_code(response)
        outcome = extract_outcome(response)
        note = extract_note(response)

        approval_limit, copay_maximum = parse_row(response)

        # Existing business rules
        if note == "1680 " and eligibility_class is None:
            eligibility_class = "out-network"

        if note == "1658 " and eligibility_class is None:
            eligibility_class = "not-active"

        return {
            "class": eligibility_class,
            "outcome": outcome,
            "note": note,
            "approval_limit": approval_limit,
            "copay_maximum": copay_maximum,
        }

    def process_visit(self, visit_id: int) -> dict:
        """
        Full workflow
        """

        # 1. Fetch visit data
        row = self.fetch_visit_data(visit_id)

        # 2. Apply transformations
        row = self.prepare_row(row)

        # 3. Generate request payload
        payload = self.generate_payload(row)

        # 4. Send API request
        response = self.call_eligibility_api(payload)

        # 5. Parse response
        parsed = self.parse_response(response)

        # 6. Final output
        return {
            "visit_id": visit_id,
            "request_payload": payload,
            "response": response,
            "parsed_response": parsed
        }

if __name__ == "__main__":

    service = EligibilityService()

    result = service.process_visit(882664)

    print(result["parsed_response"])