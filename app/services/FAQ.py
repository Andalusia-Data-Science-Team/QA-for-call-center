import os

import msal
import requests
from urllib.parse import urlsplit

SITE_URL = os.getenv(
    "SP_SITE_URL",
    "https://andalusiagroupegypt.sharepoint.com/sites/AHJ/Medical",
).rstrip("/")
TENANT = os.getenv("SP_TENANT", "")
CLIENT_ID = os.getenv("SP_CLIENT_ID", "")
USERNAME = os.getenv("SP_USERNAME", "")  # UPN, e.g. user@contoso.com
PASSWORD = os.getenv("SP_PASSWORD", "")
LIST_NAME = os.getenv("SP_LIST_NAME", "FAQ")


def require_configuration() -> None:
    missing = [
        name
        for name, value in {
            "SP_TENANT": TENANT,
            "SP_CLIENT_ID": CLIENT_ID,
            "SP_USERNAME": USERNAME,
            "SP_PASSWORD": PASSWORD,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError("Missing SharePoint Online configuration: " + ", ".join(missing))


def get_access_token() -> str:
    require_configuration()
    authority = f"https://login.microsoftonline.com/{TENANT}"
    site_parts = urlsplit(SITE_URL)
    scopes = [f"{site_parts.scheme}://{site_parts.netloc}/.default"]
    app = msal.PublicClientApplication(CLIENT_ID, authority=authority)
    result = app.acquire_token_by_username_password(
        username=USERNAME,
        password=PASSWORD,
        scopes=scopes,
    )
    token = result.get("access_token")
    if not token:
        message = result.get("error_description", result.get("error", "unknown error"))
        raise RuntimeError(f"Entra ID token request failed: {message}")
    return token


def get_list_items() -> list[dict]:
    token = get_access_token()
    list_title = LIST_NAME.replace(chr(39), chr(39) * 2)
    url = "{}/_api/web/lists/getbytitle({}{}{})/items".format(
        SITE_URL, chr(39), list_title, chr(39)
    )
    response = requests.get(
        url,
        headers={
            "Accept": "application/json;odata=nometadata",
            "Authorization": f"Bearer {token}",
        },
        timeout=30,
        verify=True,
    )
    try:
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        content_type = response.headers.get("Content-Type", "unknown")
        preview = response.text[:300].replace("\n", " ")
        raise RuntimeError(
            f"SharePoint returned non-JSON content (HTTP {response.status_code}, "
            f"Content-Type {content_type}): {preview}"
        ) from exc
    except requests.HTTPError as exc:
        content_type = response.headers.get("Content-Type", "unknown")
        preview = response.text[:300].replace("\n", " ")
        raise RuntimeError(
            f"SharePoint request failed (HTTP {response.status_code}, "
            f"Content-Type {content_type}): {preview}"
        ) from exc
    return payload.get("value", payload.get("d", {}).get("results", []))


if __name__ == "__main__":
    for item in get_list_items():
        print(item.get("Title", ""))
