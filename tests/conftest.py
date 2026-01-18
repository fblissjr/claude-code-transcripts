"""Pytest configuration and fixtures for claude-code-transcripts tests."""

import pytest
import webbrowser


@pytest.fixture(autouse=True)
def mock_webbrowser_open(monkeypatch):
    """Automatically mock webbrowser.open to prevent browsers opening during tests."""
    opened_urls = []

    def mock_open(url):
        opened_urls.append(url)
        return True

    # Patch the stdlib webbrowser.open directly
    monkeypatch.setattr(webbrowser, "open", mock_open)
    return opened_urls
