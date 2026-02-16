"""Run backtest and launch visualization dashboard.

Usage:
    python web/serve.py [--port 8080] [--force]

Flags:
    --port N    Port to serve on (default 8080)
    --force     Re-run backtest even if cached JSON exists
"""

import http.server
import os
import sys
import webbrowser
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Set environment variables before importing config
os.environ.setdefault("AMBER_API_KEY", "psk_test")
os.environ.setdefault("SOLCAST_API_KEY", "test")
os.environ.setdefault("SOLCAST_RESOURCE_ID", "test")
os.environ.setdefault("FOXESS_IP", "127.0.0.1")

WEB_DIR = Path(__file__).parent
DATA_FILE = WEB_DIR / "backtest_data.json"


def ensure_data(force: bool = False):
    """Generate backtest data if it doesn't exist."""
    if DATA_FILE.exists() and not force:
        size_mb = DATA_FILE.stat().st_size / (1024 * 1024)
        print(f"Using cached backtest data ({size_mb:.1f} MB)")
        print("  (use --force to regenerate)")
        return

    from backtest.export import export_backtest_json
    export_backtest_json(str(DATA_FILE))


def serve(port: int = 8080):
    """Start HTTP server serving the web directory."""
    os.chdir(WEB_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(("", port), handler)
    url = f"http://localhost:{port}"
    print(f"\nDashboard ready at {url}")
    print("Press Ctrl+C to stop\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    port = 8080
    force = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--force":
            force = True
            i += 1
        else:
            i += 1

    ensure_data(force)
    serve(port)
