"""Tests for repo extraction, enrichment, filtering, and display formatting."""

from ccutils.parsers.session import extract_repo_from_session
from ccutils.api import enrich_sessions_with_repos, filter_sessions_by_repo
from ccutils.cli.utils import format_session_for_display


class TestExtractRepoFromSession:
    def test_extract_from_outcomes(self):
        """Extract repo from session_context.outcomes git_info."""
        session = {
            "session_context": {
                "outcomes": [
                    {
                        "type": "git_repository",
                        "git_info": {"repo": "owner/repo-name", "type": "github"},
                    }
                ]
            }
        }
        assert extract_repo_from_session(session) == "owner/repo-name"

    def test_extract_from_sources_url(self):
        """Fall back to parsing repo from session_context.sources URL."""
        session = {
            "session_context": {
                "sources": [
                    {
                        "type": "git_repository",
                        "url": "https://github.com/owner/repo-name",
                    }
                ]
            }
        }
        assert extract_repo_from_session(session) == "owner/repo-name"

    def test_extract_from_sources_url_with_dotgit(self):
        """Handle .git suffix in source URLs."""
        session = {
            "session_context": {
                "sources": [
                    {
                        "type": "git_repository",
                        "url": "https://github.com/owner/repo-name.git",
                    }
                ]
            }
        }
        assert extract_repo_from_session(session) == "owner/repo-name"

    def test_no_context_returns_none(self):
        """Return None when no session_context exists."""
        session = {"id": "sess1", "title": "No context"}
        assert extract_repo_from_session(session) is None

    def test_empty_context_returns_none(self):
        """Return None when session_context is empty."""
        session = {"session_context": {}}
        assert extract_repo_from_session(session) is None

    def test_outcomes_preferred_over_sources(self):
        """Outcomes should be checked before sources."""
        session = {
            "session_context": {
                "outcomes": [
                    {
                        "type": "git_repository",
                        "git_info": {"repo": "from/outcomes", "type": "github"},
                    }
                ],
                "sources": [
                    {
                        "type": "git_repository",
                        "url": "https://github.com/from/sources",
                    }
                ],
            }
        }
        assert extract_repo_from_session(session) == "from/outcomes"


class TestEnrichSessionsWithRepos:
    def test_enriches_sessions(self):
        """Add repo key to each session dict."""
        sessions = [
            {
                "id": "sess1",
                "title": "Session 1",
                "created_at": "2025-01-01T10:00:00Z",
                "session_context": {
                    "outcomes": [
                        {
                            "type": "git_repository",
                            "git_info": {"repo": "owner/repo", "type": "github"},
                        }
                    ]
                },
            },
            {
                "id": "sess2",
                "title": "Session 2",
                "created_at": "2025-01-02T10:00:00Z",
                "session_context": {},
            },
        ]
        enriched = enrich_sessions_with_repos(sessions)
        assert enriched[0]["repo"] == "owner/repo"
        assert enriched[1]["repo"] is None

    def test_does_not_mutate_original(self):
        """Enrichment should not modify the original session dicts."""
        sessions = [{"id": "sess1", "session_context": {}}]
        enriched = enrich_sessions_with_repos(sessions)
        assert "repo" not in sessions[0]
        assert "repo" in enriched[0]


class TestFilterSessionsByRepo:
    def test_filters_by_repo(self):
        """Only return sessions matching the specified repo."""
        sessions = [
            {"id": "sess1", "repo": "owner/repo-a"},
            {"id": "sess2", "repo": "owner/repo-b"},
            {"id": "sess3", "repo": None},
        ]
        filtered = filter_sessions_by_repo(sessions, "owner/repo-a")
        assert len(filtered) == 1
        assert filtered[0]["id"] == "sess1"

    def test_none_repo_returns_all(self):
        """Passing None as repo returns all sessions."""
        sessions = [
            {"id": "sess1", "repo": "owner/repo-a"},
            {"id": "sess2", "repo": None},
        ]
        filtered = filter_sessions_by_repo(sessions, None)
        assert len(filtered) == 2

    def test_no_match_returns_empty(self):
        """Return empty list when no sessions match."""
        sessions = [
            {"id": "sess1", "repo": "owner/repo-a"},
        ]
        filtered = filter_sessions_by_repo(sessions, "other/repo")
        assert len(filtered) == 0


class TestFormatSessionForDisplay:
    def test_with_repo(self):
        """Display should show repo, date, and title."""
        session = {
            "id": "sess1",
            "title": "Fix the bug",
            "created_at": "2025-01-15T10:30:00.000Z",
            "repo": "owner/repo",
        }
        display = format_session_for_display(session)
        assert display.startswith("owner/repo")
        assert "2025-01-15T10:30:00" in display
        assert "Fix the bug" in display

    def test_without_repo(self):
        """Display should show (no repo) placeholder when repo is None."""
        session = {
            "id": "sess1",
            "title": "Fix the bug",
            "created_at": "2025-01-15T10:30:00.000Z",
            "repo": None,
        }
        display = format_session_for_display(session)
        assert "(no repo)" in display
        assert "Fix the bug" in display

    def test_long_title_truncated(self):
        """Titles longer than 50 chars should be truncated."""
        session = {
            "id": "sess1",
            "title": "A" * 60,
            "created_at": "2025-01-15T10:30:00.000Z",
            "repo": "owner/repo",
        }
        display = format_session_for_display(session)
        assert "..." in display
        assert "A" * 60 not in display

    def test_missing_created_at(self):
        """Handle missing created_at gracefully."""
        session = {
            "id": "sess1",
            "title": "Fix the bug",
            "repo": "owner/repo",
        }
        display = format_session_for_display(session)
        assert "N/A" in display
