import json
import logging
import urllib.parse
from sqlalchemy import create_engine, text
import pandas as pd
from arabic_reshaper import reshape
from bidi.algorithm import get_display

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_passcodes(filepath="Passcode.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["DB_NAMES"]["LIVE"]


def get_conn_engine(passcodes, logger=None):
    """
    Creates and returns a SQLAlchemy engine for connecting to the SQL database.

    Args:
    - passcodes (dict): A dictionary containing database credentials.

    Returns:
    - engine (sqlalchemy.engine.Engine): A SQLAlchemy Engine instance for connecting to the database.
    """
    try:
        server, db, uid, pwd, driver = (
            passcodes["Server"],
            passcodes["Database"],
            passcodes["UID"],
            passcodes["PWD"],
            passcodes["driver"],
        )
        params = urllib.parse.quote_plus(
            f"DRIVER={driver};"
            f"SERVER={server};"
            f"DATABASE={db};"
            f"UID={uid};"
            f"PWD={pwd};"
            f"Connection Timeout=300;"
        )
        engine = create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
        logger.debug(f"Database connection engine created for {server}/{db}")
        return engine
    except KeyError as e:
        logger.error(f"Missing key in passcodes dictionary: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error creating database connection engine: {e}")
        raise

def reshape_text(value):
    if isinstance(value, str):
        return get_display(reshape(value))
    return value


def reshape_dataframe(df):
    """Reshape all Arabic column labels and cell values in the DataFrame."""
    df = df.copy()
    df.columns = [reshape_text(col) for col in df.columns]
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].apply(reshape_text)
    return df

if __name__ == "__main__":
    passcodes = load_passcodes("/home/ai/Workspace/Rafik/QA_System-main/app/Passcode.json")
    print(passcodes)
    engine = get_conn_engine(passcodes, logger=logger)
    print("Database connection engine created successfully.")
    with open("/home/ai/Workspace/Rafik/QA_System-main/app/SQL/Slots.sql", "r") as f:
        sql_query = f.read()
    query = text(sql_query)
    df = pd.read_sql(query, engine, params={"ReportDate": "2026-02-05", "Specialty": "أمراض الذكورة", "Doctor": "شريف الضوى "})
    df = reshape_dataframe(df)
    print(df)

