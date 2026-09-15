import ast
import json
import platform
import random
import time
import warnings
from datetime import datetime

try:
    import cx_Oracle
except ModuleNotFoundError:  # Oracle support is only needed for ETL client setup.
    cx_Oracle = None
import pandas as pd
import requests
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # Only required by the bulk Iqama_table workflow.
    tqdm = None
import pyodbc

# Suppress the specific UserWarning
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Could not infer format"
)


def update_table(passcodes, table_name, df, logger, retries=5, delay=30):
    """
    Updates a database table with manual SQL INSERT statements.
    Completely bypasses df.to_sql() to avoid cursor attribute errors.
    """
    server = passcodes["Server"]
    if "," not in server and "\\" not in server:
        server = f"{server},1433"

    connection_string = (
        f"DRIVER={passcodes['driver']};"
        f"SERVER={server};"
        f"DATABASE={passcodes['Database']};"
        f"UID={passcodes['UID']};"
        f"PWD={passcodes['PWD']};"
        f"Connection Timeout=60;"
        f"Login Timeout=60;"
        f"Encrypt=no;"
        f"TrustServerCertificate=yes;"
    )

    # Create a copy of the DataFrame
    df_clean = df.copy()

    # Format insertion_date properly
    if "insertion_date" in df_clean.columns:
        df_clean["insertion_date"] = pd.to_datetime(
            df_clean["insertion_date"]
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Replace NaN with None for SQL NULL
    df_clean = df_clean.where(pd.notnull(df_clean), None)

    attempt = 0
    while attempt < retries:
        conn = None
        try:
            # Connect using pyodbc
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()

            # Test connection
            cursor.execute("SELECT GETDATE()")
            server_time = cursor.fetchone()[0]
            logger.info(f"Connection test successful. Server time: {server_time}")

            # Prepare column names and placeholders
            columns = list(df_clean.columns)
            column_names = ", ".join(
                [f"[{col}]" for col in columns]
            )  # Bracket column names for safety
            placeholders = ", ".join(["?" for _ in columns])

            # Create INSERT statement
            insert_sql = f"INSERT INTO dbo.[{table_name}] ({column_names}) VALUES ({placeholders})"
            logger.info(f"INSERT SQL: {insert_sql}")

            # Insert data in chunks
            chunk_size = 1000
            total_rows = len(df_clean)
            inserted_count = 0

            for i in range(0, total_rows, chunk_size):
                chunk = df_clean.iloc[i: i + chunk_size]

                # Convert DataFrame chunk to list of tuples
                data_tuples = []
                for _, row in chunk.iterrows():
                    # Convert each row to tuple, handling special data types
                    row_tuple = []
                    for value in row:
                        if pd.isna(value):
                            row_tuple.append(None)
                        elif isinstance(value, (pd.Timestamp, datetime)):
                            row_tuple.append(value.strftime("%Y-%m-%d %H:%M:%S"))
                        else:
                            row_tuple.append(value)
                    data_tuples.append(tuple(row_tuple))

                # Execute batch insert
                cursor.executemany(insert_sql, data_tuples)
                conn.commit()

                inserted_count += len(data_tuples)
                logger.info(
                    f"Inserted chunk {i//chunk_size + 1}: {len(data_tuples)} rows (Total: {inserted_count}/{total_rows})"
                )

            cursor.close()
            logger.info(
                f"Table {table_name} updated successfully with {inserted_count} records"
            )
            return

        except Exception as e:
            attempt += 1
            logger.error(f"Attempt {attempt} failed: {e}")

            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass

            if attempt < retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("All retries failed.")
                raise

        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass


def Beneficiary_api(patient_iqama_id):
    # Define the API URL
    url = (
        f"http://10.111.111.6:8000/check-insurance?patient_iqama_id={patient_iqama_id}"
    )
    # Send a GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()

        return data
    else:
        print(f"Error: {response.status_code}, {response.text}")


def name_conflict(df):
    column_pairs = [
        # Example pairs - replace with your actual column names
        ("InsuranceCompanyAR", "ArabicICName"),
        ("InsuranceCompanyEN", "InsuranceCompanyName"),
        ("IssueDate", "ISSUEDATE"),
        ("Name", "NAME"),
        ("IdentityNumber", "IDENTITYNUMBER"),
    ]

    for primary_col, alt_col in column_pairs:
        if primary_col in df.columns and alt_col in df.columns:
            # Fill null values in primary column with values from alternative column
            df[primary_col] = df[primary_col].fillna(df[alt_col])

            # Optionally, drop the alternative column if not needed
            df.drop(columns=[alt_col], inplace=True)

    return df


# Function to safely parse the JSON/dict data and extract insurance info
def extract_insurance_data(row):
    try:
        # Try different parsing methods
        data = None
        if isinstance(row, dict):
            data = row
        elif isinstance(row, str):
            try:
                data = json.loads(row)
            except Exception:
                try:
                    data = ast.literal_eval(row)
                except Exception:
                    return None

        # Check if ApiStatus is Success and extract Insurance data
        if data and data.get("response", {}).get("ApiStatus") == "Success":
            insurance_list = data.get("response", {}).get("Insurance", [])
            if insurance_list and len(insurance_list) > 0:
                # Return the first insurance entry
                return insurance_list[0]
        return None
    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def Iqama_table(df, logger):
    """
    1- Gets unique int iqama numbers from extracted df
    2- Sends them to Beneficiary API with a random delay using Beneficiary_api()
    3- Parses the resulting JSON and extracts success/failure status, and insurance data of each request,
      using extract_api_status() and extract_insurance_data()

    """
    insurance_df = pd.DataFrame()
    # Get unique iqama numbers
    unique_iqama = df["iqama_no"].dropna().unique()
    logger.info(f"length of iqama_pateinet_id:  {len(unique_iqama)}")

    # Apply the API to unique iqama numbers and store in a dictionary
    eligibility_dict = {}
    for iqama in unique_iqama:
        try:
            iqama = int(iqama)
            eligibility_dict[int(iqama)] = Beneficiary_api(int(iqama))
            # Random delay between 10 and 20 milliseconds
            delay_ms = random.uniform(10, 20)
            time.sleep(delay_ms / 1000)
        except ValueError:
            iqama = None

    # Convert to DataFrame
    _df = pd.DataFrame(eligibility_dict.items(), columns=["Iqama_no", "Eligibility"])

    logger.info("Extracting API status...")
    if tqdm is None:
        raise RuntimeError("tqdm is required for bulk eligibility processing")
    tqdm.pandas(desc="Extracting API Status")
    _df["api_status"] = _df["Eligibility"].progress_apply(extract_api_status)

    len_success = len(_df[_df["api_status"] == "Success"])
    len_fail = len(_df[_df["api_status"] == "Fail"])

    logger.info(f"length of API-Call Successed: {len_success}")
    logger.info(f"length of API-Call Failed: {len_fail}")

    # Extract insurance data
    insurance_data = _df["Eligibility"].apply(extract_insurance_data)
    insurance_df = insurance_data.apply(pd.Series)
    insurance_df = pd.concat([_df, insurance_df], axis=1)

    columns_to_drop = ["UploadDate", "NationalityCode", "Eligibility", "api_status"]
    existing_columns = [col for col in columns_to_drop if col in insurance_df.columns]
    insurance_df.drop(columns=existing_columns, inplace=True)
    insurance_df = name_conflict(insurance_df)
    insurance_df.dropna(inplace=True)

    df["iqama_no"] = pd.to_numeric(df["iqama_no"], errors="coerce").astype("Int64")
    insurance_df["Iqama_no"] = pd.to_numeric(
        insurance_df["Iqama_no"], errors="coerce"
    ).astype("Int64")
    insurance_df = insurance_df.merge(
        df[["iqama_no", "visit_id"]],
        left_on="Iqama_no",
        right_on="iqama_no",
        how="left",
    )
    insurance_df.drop(columns=["iqama_no"], inplace=True)

    return insurance_df


def extract_api_status(value):
    try:
        # Try parsing as JSON first
        try:
            value = json.dumps(value)
            data = json.loads(value)
        except json.JSONDecodeError:
            # If that fails, try evaluating it as a Python literal
            try:
                data = ast.literal_eval(value)
            except Exception:
                # Print a sample of the problematic value for debugging
                print(f"Cannot parse: {value[:50]}...")
                return None

        # Extract ApiStatus from the parsed data
        if isinstance(data, dict):
            response = data.get("response", {})
            if isinstance(response, dict):
                return response.get("ApiStatus")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None


def change_date(row):
    if pd.isna(row):
        return None

    row = str(row).split(".")[0]  # Remove anything after a dot (like ".357000")

    try:
        # Try parsing with time first
        date_object = datetime.strptime(row, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            # If that fails, try parsing date only
            date_object = datetime.strptime(row, "%Y-%m-%d")
        except ValueError:
            # If it still fails, return None (or raise an error if you prefer)
            return None

    return date_object.strftime("%Y-%m-%d")


def create_json_payload(row, purpose="discovery", source=""):
    # Helper function to safely convert values to int and then string
    def safe_int_str(value, default=""):
        if pd.notna(value) and value is not None:
            try:
                return str(int(value))
            except (ValueError, TypeError):
                return default
        return default

    # Helper function to safely convert values to string
    def safe_str(value, default=""):
        if pd.notna(value) and value is not None:
            return str(value)
        return default

    # Helper function to safely handle patient given names
    def safe_name_list(name1, name2):
        result = []
        if pd.notna(name1) and name1 is not None:
            result.append(name1)
        if pd.notna(name2) and name2 is not None:
            result.append(name2)
        return result

    # Determine identifier system safely
    identifier_system = (
        "nationalid" if safe_str(row.get("nationality")) == "NI" else "iqama"
    )
    if source == "Replica":
        json_data = {
            "purpose": purpose,
            "patient_id": safe_int_str(row.get("patient_id")),
            "payer_license": safe_int_str(row.get("payer_linces")),
            "payer_license_2": safe_int_str(row.get("payer_linces")),
            "serviced_period_start": safe_str(row.get("start_date")),
            "serviced_period_end": safe_str(row.get("end_date")),
            "created_date": safe_str(row.get("start_date")),
            "patient_name": safe_str(row.get("patient_name")),
            "patient_family_name": safe_str(row.get("family_name")),
            "patient_given_names": safe_name_list(
                row.get("pat_name_1"), row.get("pat_name_2")
            ),
            "patient_gender": safe_str(row.get("gender")),
            "patient_birth_date": safe_str(row.get("date_of_birth")),
            "marital_status_code": safe_str(row.get("marital_char")),
            "patient_identifier_value": safe_int_str(row.get("iqama_no")),
            "patient_identifier_type": safe_str(row.get("nationality")),
            "patient_identifier_system": identifier_system,
            "patient_occupation_code": "others",
            "insurer_name": safe_str(row.get("purchaser_name")),
            "insurer_org_id": safe_int_str(row.get("insurer")),
        }
    else:
        json_data = {
            "purpose": purpose,
            "patient_id": safe_int_str(row.get("patient_id")),
            "payer_license": safe_int_str(row.get("payer_linces")),
            "payer_license_2": safe_int_str(row.get("payer_linces")),
            "serviced_period_start": safe_str(row.get("start_date")),
            "serviced_period_end": safe_str(row.get("end_date")),
            "created_date": safe_str(row.get("start_date")),
            "patient_name": safe_str(row.get("patient_name")),
            "patient_family_name": safe_str(row.get("family_name")),
            "patient_given_names": safe_name_list(
                row.get("pat_name_1"), row.get("pat_name_2")
            ),
            "patient_gender": safe_str(row.get("gender")),
            "patient_birth_date": safe_str(row.get("date_of_birth")),
            "marital_status_code": safe_str(row.get("marital_char")),
            "patient_identifier_value": safe_int_str(row.get("iqama_no")),
            "patient_identifier_type": safe_str(row.get("nationality")),
            "patient_identifier_system": identifier_system,
            "patient_occupation_code": "others",
            "insurer_name": safe_str(row.get("purchaser_name")),
            "insurer_org_id": safe_int_str(row.get("insurer")),
        }

    return json_data


def send_json_to_api(json_data):
    api_url = "http://10.111.111.6:8001/submit_eligibility"
    headers = {"Content-Type": "application/fhir+json"}
    try:
        response = requests.post(api_url, json=json_data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def extract_outcome(response):
    # Handle None responses
    if response is None:
        # print("Response is None. Cannot extract outcome.")
        return "Null"

    # Check if the status is "success"
    if response.get("status") != "success":
        # print("Status is not 'success'. Cannot proceed.")
        return "Null"

    # Safely get entries
    entries = response.get("response", {}).get("entry", [])
    if not entries:
        print("No entries found in the response.")
        return "Null"

    # Extract the value of the "outcome" key
    outcome_value = None

    for entry in entries:
        resource = entry.get("resource", {})
        outcome_value = resource.get("outcome")
        if outcome_value is not None:
            break

    if outcome_value == "complete":
        return "Complete"
    elif outcome_value == "error":
        return "ERROR"
    else:
        return "Null"


def extract_code(response):
    # Handle None responses
    if response is None:
        print("Response is None. Cannot proceed.")
        return None

    # Check if the status is "success"
    if response.get("status") != "success":
        print("Status is not 'success'. Cannot proceed.")
        return None

    # Check if the response has the expected structure
    if not isinstance(response.get("response"), dict):
        print(
            "Response format is not as expected. Missing or invalid 'response' field."
        )
        return None

    # Check if 'entry' exists in the response
    entries = response.get("response", {}).get("entry", [])
    if not entries:
        print("No entries found in the response.")
        return None

    # Loop through the entries to find CoverageEligibilityResponse
    for entry in entries:
        resource = entry.get("resource", {})
        if (
            resource.get("resourceType") == "CoverageEligibilityResponse"
            and resource.get("outcome") == "complete"
        ):
            for ext in resource.get("extension", []):
                if (
                    ext.get("url")
                    == "http://nphies.sa/fhir/ksa/nphies-fs/StructureDefinition/extension-siteEligibility"
                ):
                    coding = ext.get("valueCodeableConcept", {}).get("coding", [])
                    for code_item in coding:
                        code_value = code_item.get("code")
                        return code_value

    return None


def extract_note(response):
    disposition_value = ""

    # Handle None responses
    if response is None:
        print("Response is None. Cannot extract note.")
        return None

    # Check if the status is "success"
    if response.get("status") != "success":
        print("Status is not 'success'. Cannot proceed.")
        return None

    # Safely get entries
    entries = response.get("response", {}).get("entry", [])
    if not entries:
        print("No entries found in the response.")
        return None

    # Extract the value of the "outcome" key
    outcome_value = None
    error_value = None

    for entry in entries:
        resource = entry.get("resource", {})
        outcome_value = resource.get("outcome")

        if outcome_value == "error":
            error_value = resource.get("error")

        if outcome_value == "complete":
            disposition_value = resource.get("disposition", "")

        if outcome_value is not None:
            break

    # Print the outcome value
    if outcome_value is None:
        print("Outcome key not found in the response.")

    # If the outcome is "error", return error details
    if outcome_value == "error":
        if error_value is not None:
            for error in error_value:
                code_obj = error.get("code", {})
                coding_list = code_obj.get("coding", [])
                for coding in coding_list:
                    return f"{coding.get('code', '')} {coding.get('display', '')}"
        else:
            print("Error key not found in the resource.")
            return "Unknown error"

    # Return disposition value for complete outcomes
    elif disposition_value:
        return disposition_value
    else:
        print("Disposition key not found in the resource.")
        return None


def find_keys(data, target="allowedMoney"):
    results = []

    if isinstance(data, dict):
        for k, v in data.items():
            if k == target:
                results.append(v)
            results.extend(find_keys(v, target))
    elif isinstance(data, list):
        for item in data:
            results.extend(find_keys(item, target))

    return results


def extract_allowedMoney(results):
    # Works with Tawuniya jsons
    if not results or len(results) < 2:
        return None, None
    return results[0]["value"], results[1]["value"]


def extract_maxcopay(results):
    # Works with malath jsons
    try:
        return None, results[0][1]["valueMoney"]["value"]
    except Exception:
        return None, None


def bupa_approval_limit(json_obj):
    # Works with bupa jsons
    try:
        items = json_obj["response"]["entry"][1]["resource"]["insurance"][0]["item"]
    except Exception:
        return None, None

    for it in items:
        if it.get("name") == "Approval limit":
            # the benefit list always contains allowedMoney as first element
            ben = it.get("benefit", [])
            for b in ben:
                if "allowedMoney" in b:
                    return b["allowedMoney"]["value"], None

    return None, None


def parse_row(x):
    s = str(x)

    if "tawuniya.com.sa" in s:
        keys = find_keys(x)
        return extract_allowedMoney(keys)

    elif "www.malath.com.sa" in s:
        keys = find_keys(x, "costToBeneficiary")
        return extract_maxcopay(keys)

    elif "bupa.com.sa" in s:
        return bupa_approval_limit(x)

    else:
        return None, None


def init_oracle_client(lib_dir=None):
    # The lib_dir parameter is important only for Windows platform.
    if cx_Oracle is None:
        raise RuntimeError("cx_Oracle is required to initialize the Oracle client")

    if platform.system() == "Windows":
        oracle_dir = lib_dir
    else:
        oracle_dir = None
    try:

        cx_Oracle.init_oracle_client(oracle_dir)
        print("Oracle Client initialized successfully.")
    except cx_Oracle.ProgrammingError as e:
        if "already been initialized" in str(e):
            print("Oracle Client library has already been initialized.")
        else:
            raise Exception


def map_row(row):
    gender_map = {"Female": "female", "Male": "male"}
    marital_map = {"Single": "U", "Married": "M"}

    row["gender"] = gender_map.get(row["gender"], row["gender"])
    row["marital_char"] = marital_map.get(row["marital_char"], row["marital_char"])

    return row
