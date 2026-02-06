"""Claude API client and credentials management.

This module provides functions for authenticating with and fetching data
from the Claude API.
"""

import json
import os
import platform
import subprocess
from pathlib import Path

import httpx

from ..parsers.session import extract_repo_from_session

# API constants
API_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class CredentialsError(Exception):
    """Raised when credentials cannot be obtained."""

    pass


def get_access_token_from_keychain():
    """Get access token from macOS keychain.

    Returns the access token or None if not found.
    Raises CredentialsError with helpful message on failure.
    """
    if platform.system() != "Darwin":
        return None

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                os.environ.get("USER", ""),
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None

        # Parse the JSON to get the access token
        creds = json.loads(result.stdout.strip())
        return creds.get("claudeAiOauth", {}).get("accessToken")
    except (json.JSONDecodeError, subprocess.SubprocessError):
        return None


def get_org_uuid_from_config():
    """Get organization UUID from ~/.claude.json.

    Returns the organization UUID or None if not found.
    """
    config_path = Path.home() / ".claude.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("oauthAccount", {}).get("organizationUuid")
    except (json.JSONDecodeError, IOError):
        return None


def get_api_headers(token, org_uuid):
    """Build API request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
        "x-organization-uuid": org_uuid,
    }


def fetch_sessions(token, org_uuid, debug=False, limit=None):
    """Fetch list of sessions from the API with pagination support.

    Handles pagination by following `has_more` and using `last_id` as cursor.
    Returns the sessions data as a dict with all sessions combined in "data".
    Raises httpx.HTTPError on network/API errors.

    Args:
        token: API access token
        org_uuid: Organization UUID
        debug: If True, returns after first page for inspection
        limit: Optional limit per page (useful for debugging API behavior)
    """
    headers = get_api_headers(token, org_uuid)
    all_sessions = []
    after_id = None

    while True:
        # Build params fresh each iteration
        params = {}
        if limit:
            params["limit"] = limit
        if after_id:
            params["after_id"] = after_id

        response = httpx.get(
            f"{API_BASE_URL}/sessions",
            headers=headers,
            params=params if params else None,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        sessions = data.get("data", [])
        all_sessions.extend(sessions)

        # In debug mode, return the raw first page response for inspection
        if debug:
            return data

        # Check for more pages using has_more and last_id (common API pattern)
        has_more = data.get("has_more", False)
        last_id = data.get("last_id")

        if not has_more or not last_id:
            break

        # Use last_id as cursor for next page
        after_id = last_id

    # Return combined result with same structure as single-page response
    return {"data": all_sessions, "has_more": False}


def fetch_session(token, org_uuid, session_id):
    """Fetch a specific session from the API.

    Returns the session data as a dict.
    Raises httpx.HTTPError on network/API errors.
    """
    headers = get_api_headers(token, org_uuid)
    response = httpx.get(
        f"{API_BASE_URL}/session_ingress/session/{session_id}",
        headers=headers,
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def enrich_sessions_with_repos(sessions):
    """Enrich sessions with repo information from session metadata.

    Args:
        sessions: List of session dicts from the API

    Returns:
        List of session dicts with 'repo' key added
    """
    enriched = []
    for session in sessions:
        session_copy = dict(session)
        session_copy["repo"] = extract_repo_from_session(session)
        enriched.append(session_copy)
    return enriched


def filter_sessions_by_repo(sessions, repo):
    """Filter sessions by repo.

    Args:
        sessions: List of session dicts with 'repo' key
        repo: Repo to filter by (owner/name), or None to return all

    Returns:
        Filtered list of sessions
    """
    if repo is None:
        return sessions
    return [s for s in sessions if s.get("repo") == repo]


__all__ = [
    "API_BASE_URL",
    "ANTHROPIC_VERSION",
    "CredentialsError",
    "get_access_token_from_keychain",
    "get_org_uuid_from_config",
    "get_api_headers",
    "fetch_sessions",
    "fetch_session",
    "enrich_sessions_with_repos",
    "filter_sessions_by_repo",
]
