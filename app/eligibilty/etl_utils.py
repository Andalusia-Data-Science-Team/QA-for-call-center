import smtplib
import time
import urllib.parse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import pendulum
from airflow.utils.email import send_email
from sqlalchemy import create_engine
from sqlalchemy import text

email_list = [
    "Doha.Ramadan@Andalusiagroup.net",
    "Nadine.ElSokily@Andalusiagroup.net",
    "Aya.Ramadan@Andalusiagroup.net"
]
START_DATE = pendulum.now(pendulum.timezone("Africa/Cairo")).subtract(days=1)


def get_conn_engine(passcodes, logger):
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


def read_data(query, passcode, logger):
    """
    Executes a SQL query using get_conn_engine. If the first attempt fails, waits for 5 minutes before trying again.

    Returns:
        pandas DataFrame with query results
    """
    try:
        df = pd.read_sql_query(query, get_conn_engine(passcode, logger))
        logger.info(f"Query returned dataframe with {len(df)} rows")
        return df
    except Exception as e:
        logger.debug(f"First attempt failed with error: {str(e)}")
        logger.debug("Waiting 5 minutes before retrying...")
        time.sleep(300)  # Wait for 5 minutes (300 seconds)

        # Second attempt
        try:
            return pd.read_sql_query(query, get_conn_engine(passcode, logger))
        except Exception as e:
            logger.debug(f"Data extraction attempt failed with error: {str(e)}")
            logger.exception("Second attempt to execute read_sql_query failed")
            # Raise so callers can handle retry/failure properly instead of receiving None
            raise





def update_table_KSA(passcode, table_name, df, logger, retries=3, delay=180):
    try:
        engine = get_conn_engine(passcode, logger)
        df_clean = df.copy()
        
        # EXACT TABLE NAMES PROVIDED
        staging_table = "[dbo].[AHJ_Medical_Prediction_Final_Test]"
        final_table = "DWH_Claims.dbo.AHJ_Medical_Prediction_Final"

        attempt = 0
        while attempt < retries:
            try:
                logger.debug(f"Upsert attempt {attempt+1}/{retries}")

                # STEP 1: Load data into the staging table
                # We use 'replace' to ensure the staging table only holds the current batch
                df_clean.to_sql(
                    name="AHJ_Medical_Prediction_Final_Test", # Name without brackets for pandas
                    con=engine,
                    index=False,
                    if_exists="replace",
                    schema="dbo"
                )

                # STEP 2: Construct the MERGE SQL
                # We wrap columns in [] to handle the '/' in 'Reason/Recommendation'
                cols = [f"[{col}]" for col in df_clean.columns]
                update_cols = [f"target.[{col}] = source.[{col}]" for col in df_clean.columns if col != 'VisitServiceID']
                
                merge_sql = f"""
                MERGE INTO {final_table} AS target
                USING {staging_table} AS source
                ON target.[VisitServiceID] = source.[VisitServiceID]
                WHEN MATCHED THEN
                    UPDATE SET {", ".join(update_cols)}
                WHEN NOT MATCHED THEN
                    INSERT ({", ".join(cols)})
                    VALUES ({", ".join(['source.' + c for c in cols])});
                """

                # STEP 3: Execute the Merge inside a transaction
                with engine.begin() as conn:
                    conn.execute(text(merge_sql))
                    # Optional: Clean staging table after merge
                    conn.execute(text(f"TRUNCATE TABLE {staging_table}"))

                logger.info(f"Successfully upserted {len(df_clean)} rows into {final_table}")
                return 
                
            except Exception as e:
                attempt += 1
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                if attempt < retries:
                    time.sleep(delay)
                else:
                    raise
    except Exception as e:
        logger.exception(f"Critical error in updating table: {e}")
        raise



def update_table_EGY(passcode, table_name, df, logger, retries=3, delay=180):
    try:
        engine = get_conn_engine(passcode, logger)
        df_clean = df.copy()
        
        # EXACT TABLE NAMES PROVIDED
        staging_table = "[dbo].[EGY_MedPred_STG]"
        final_table = "DWH_Claims.dbo.EGY_MedPred_Final"

        attempt = 0
        while attempt < retries:
            try:
                logger.debug(f"Upsert attempt {attempt+1}/{retries}")

                # STEP 1: Load data into the staging table
                # We use 'replace' to ensure the staging table only holds the current batch
                df_clean.to_sql(
                    name="EGY_MedPred_STG",    # Name without brackets for pandas
                    con=engine,
                    index=False,
                    if_exists="replace",
                    schema="dbo"
                )

                # STEP 2: Construct the MERGE SQL
                # We wrap columns in [] to handle the '/' in 'Reason/Recommendation'
                cols = [f"[{col}]" for col in df_clean.columns]
                update_cols = [f"target.[{col}] = source.[{col}]" for col in df_clean.columns if col != 'VisitServiceID']
                
                merge_sql = f"""
                MERGE INTO {final_table} AS target
                USING {staging_table} AS source
                ON target.[VisitServiceID] = source.[VisitServiceID]
                WHEN MATCHED THEN
                    UPDATE SET {", ".join(update_cols)}
                WHEN NOT MATCHED THEN
                    INSERT ({", ".join(cols)})
                    VALUES ({", ".join(['source.' + c for c in cols])});
                """

                # STEP 3: Execute the Merge inside a transaction
                with engine.begin() as conn:
                    conn.execute(text(merge_sql))
                    # Optional: Clean staging table after merge
                    conn.execute(text(f"TRUNCATE TABLE {staging_table}"))

                logger.info(f"Successfully upserted {len(df_clean)} rows into {final_table}")
                return 
                
            except Exception as e:
                attempt += 1
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                if attempt < retries:
                    time.sleep(delay)
                else:
                    raise
    except Exception as e:
        logger.exception(f"Critical error in updating table: {e}")
        raise



def update_table(passcode, table_name, df, logger, retries=3, delay=180):
    """
    Updates a database table with the given DataFrame. Retries on failure.

    Parameters:
    - table_name: Name of the table to update.
    - df: DataFrame to update the table.
    - retries: Number of retry attempts.
    - delay: Delay in seconds between retries.
    """
    try:
        engine = get_conn_engine(passcode, logger)
        # Create a copy of the DataFrame to avoid modifying the original
        df_clean = df.copy()

        attempt = 0
        while attempt < retries:
            try:
                logger.debug(f"Update attempt {attempt+1}/{retries}")
                logger.debug("Connection established, beginning data transfer")
                df_clean.to_sql(
                    name=f"{table_name}",
                    con=engine,
                    index=False,
                    if_exists="append",
                    chunksize=1000,
                    schema="dbo",
                )
                logger.info(f"Successfully updated '{table_name}' table")
                return  # Exit the function if successful
            except Exception as e:
                attempt += 1
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    failure_msg = "All retries failed. Please check the error and try again later."
                    logger.error(failure_msg)
                    raise  # Re-raise the exception after all retries fail
    except Exception as e:
        logger.exception(f"Critical error in updating table {table_name}: {e}")
        raise

def failure_callback(context):
    """Enhanced failure callback with SMTP email sending"""
    try:
        dag_run = context.get("dag_run")
        task_instance = context.get("task_instance")
        exception = context.get("exception")
        execution_date = context.get("execution_date")

        # Format execution date properly
        exec_date_str = (
            execution_date.strftime("%Y-%m-%d %H:%M:%S")
            if execution_date
            else "Unknown"
        )

        subject = (
            f"[Airflow FAILURE] DAG: {dag_run.dag_id} - Task: {task_instance.task_id}"
        )

        # Create a more detailed HTML email body
        html_body = f"""
        <html>
        <body>
            <h2 style="color: #d32f2f;">Airflow DAG Failure Alert</h2>
            <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse;">
                <tr>
                    <td><strong>DAG ID:</strong></td>
                    <td>{dag_run.dag_id}</td>
                </tr>
                <tr>
                    <td><strong>Task ID:</strong></td>
                    <td>{task_instance.task_id}</td>
                </tr>
                <tr>
                    <td><strong>Execution Date:</strong></td>
                    <td>{exec_date_str}</td>
                </tr>
                <tr>
                    <td><strong>Run ID:</strong></td>
                    <td>{dag_run.run_id}</td>
                </tr>
                <tr>
                    <td><strong>State:</strong></td>
                    <td style="color: #d32f2f;">{task_instance.state}</td>
                </tr>
                <tr>
                    <td><strong>Exception:</strong></td>
                    <td><pre style="background-color: #f5f5f5; padding: 10px;">{str(exception) if exception else 'No exception details available'}</pre></td>
                </tr>
            </table>
            <br>
            <p><strong>Please investigate this failure and take appropriate action.</strong></p>
        </body>
        </html>
        """

        # Send email using SMTP
        try:
            print("Sending failure notification via SMTP...")
            server = smtplib.SMTP("aws-ex-07.andalusia.loc", 25)
            server.set_debuglevel(1)
            server.starttls()

            # Create message
            msg = MIMEMultipart()
            msg["From"] = "ai-service@andalusiagroup.net"
            msg["To"] = ", ".join(email_list)
            msg["Subject"] = subject
            msg.attach(MIMEText(html_body, "html"))

            # Send without authentication
            server.send_message(msg)
            server.quit()
            print("✅ Failure notification sent successfully!")

        except Exception as smtp_error:
            print(f"❌ Failed to send failure notification via SMTP: {smtp_error}")
            # Fall back to Airflow's send_email if SMTP fails
            try:
                send_email(
                    to=email_list,
                    subject=subject,
                    html_content=html_body,
                )
                print("Used fallback email method successfully")
            except Exception as fallback_error:
                print(f"Failed to send email with fallback method: {fallback_error}")

    except Exception as e:
        print(f"Failed to process failure notification: {str(e)}")
