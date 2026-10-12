import os
import socket
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


DEFAULT_URL = (
    "http://ahj-hmis:8090/"
    "AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue"
)
DEFAULT_IDENTIFICATION_VALUE = 1070654163


def main():
    load_dotenv()
    url = os.getenv("MPI_API_URL", DEFAULT_URL)
    token = os.getenv("MPI_AUTH_TOKEN")
    if not token:
        sys.exit("Set MPI_AUTH_TOKEN in the project .env file before running this script.")

    hostname = urlparse(url).hostname
    try:
        socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        sys.exit(
            f"Cannot resolve MPI host '{hostname}'. Connect to the corporate VPN/network, "
            "or set MPI_API_URL in .env to the correct reachable endpoint."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    headers = {
        "Referer": "http://dev-testing.andalusiagroup.net:5004/finance/receivable/opd-invoice-list",
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/json; charset=utf-8",
        "Authentication": token,
        "x-request-timestamp": timestamp,
    }
    params = {"identificationValue": os.getenv("MPI_IDENTIFICATION_VALUE", DEFAULT_IDENTIFICATION_VALUE)}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as error:
        body = error.response.text.strip()
        sys.exit(f"Request failed: {error}\nServer response: {body}")
    except requests.RequestException as error:
        sys.exit(f"Request failed: {error}")

    try:
        print(response.json())
    except requests.exceptions.JSONDecodeError:
        print(response.text)


if __name__ == "__main__":
    main()
