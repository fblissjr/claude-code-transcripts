"""CLI commands for ccutils (Claude Code utilities)."""

import click
from click_default_group import DefaultGroup

from .local import local_cmd
from .web import web_cmd
from .json_cmd import json_cmd
from .all import all_cmd
from .explore import explore_cmd
from .utils import (
    is_url,
    fetch_url_to_tempfile,
    resolve_credentials,
    format_session_for_display,
    generate_html_from_session_data,
)


@click.group(cls=DefaultGroup, default="local", default_if_no_args=True)
@click.version_option(None, "-v", "--version", package_name="ccutils")
def cli():
    """Convert Claude Code sessions to HTML pages or DuckDB databases.

    Export individual sessions to HTML, or batch export all sessions
    to a browsable HTML archive or DuckDB database for analytics.
    """
    pass


# Register commands
cli.add_command(local_cmd, "local")
cli.add_command(json_cmd, "json")
cli.add_command(web_cmd, "web")
cli.add_command(all_cmd, "all")
cli.add_command(explore_cmd, "explore")


def main():
    cli()


# Re-export for backwards compatibility
__all__ = [
    "cli",
    "main",
    "local_cmd",
    "json_cmd",
    "web_cmd",
    "all_cmd",
    "explore_cmd",
    "is_url",
    "fetch_url_to_tempfile",
    "resolve_credentials",
    "format_session_for_display",
    "generate_html_from_session_data",
]
