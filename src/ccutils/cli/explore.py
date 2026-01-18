"""Data Explorer command for analyzing DuckDB star schema databases."""

import http.server
import socketserver
import webbrowser
from pathlib import Path

import click


@click.command("explore")
@click.option(
    "-p",
    "--port",
    default=8765,
    help="Port to serve on (default: 8765).",
)
@click.option(
    "--no-open",
    is_flag=True,
    help="Don't open browser automatically.",
)
@click.argument("database", required=False, type=click.Path(exists=True))
def explore_cmd(port, no_open, database):
    """Launch the Data Explorer to analyze DuckDB star schema databases.

    Starts a local web server and opens the Data Explorer in your browser.
    Optionally specify a DATABASE file to have it ready to load.

    Examples:

        claude-code-transcripts explore

        claude-code-transcripts explore my-archive.duckdb
    """
    # Get data explorer directory
    explorer_dir = Path(__file__).parent.parent / "explorer"
    if not explorer_dir.exists():
        raise click.ClickException(f"Data explorer directory not found: {explorer_dir}")

    explorer_file = explorer_dir / "index.html"
    if not explorer_file.exists():
        raise click.ClickException(f"Data explorer not found: {explorer_file}")

    # Create a simple HTTP server with proper cleanup
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(explorer_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # Suppress request logging

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = None
    try:
        httpd = ReusableTCPServer(("", port), QuietHandler)
        url = f"http://localhost:{port}/index.html"
        click.echo(f"Data Explorer running at: {url}")
        if database:
            db_path = Path(database).resolve()
            click.echo(f"Load database: {db_path}")
        click.echo("Press Ctrl+C to stop the server.")

        if not no_open:
            webbrowser.open(url)

        httpd.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopping server...")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            raise click.ClickException(
                f"Port {port} is already in use. Try a different port with -p/--port."
            )
        raise
    finally:
        if httpd:
            httpd.shutdown()
            httpd.server_close()
            click.echo("Server stopped.")
