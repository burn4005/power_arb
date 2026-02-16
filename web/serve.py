"""Run backtest and launch visualization dashboard.

Usage:
    python web/serve.py [--port 8080] [--force]

Flags:
    --port N    Port to serve on (default 8080)
    --force     Re-run backtest even if cached JSON exists
"""

import csv
import http.server
import json
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
ENV_PATH = ROOT / ".env"


def ensure_data(force: bool = False):
    """Generate backtest data if it doesn't exist."""
    if DATA_FILE.exists() and not force:
        size_mb = DATA_FILE.stat().st_size / (1024 * 1024)
        print(f"Using cached backtest data ({size_mb:.1f} MB)")
        print("  (use --force to regenerate)")
        return

    from backtest.export import export_backtest_json
    export_backtest_json(str(DATA_FILE))


def read_env_file() -> dict[str, str]:
    settings = {}
    if ENV_PATH.exists():
        with open(ENV_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    settings[key.strip()] = value.strip()
    return settings


def write_env_file(settings: dict[str, str]):
    lines = []
    written_keys = set()
    if ENV_PATH.exists():
        with open(ENV_PATH, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.partition("=")[0].strip()
                    if key in settings:
                        lines.append(f"{key}={settings[key]}\n")
                        written_keys.add(key)
                    else:
                        lines.append(line)
                else:
                    lines.append(line)
    new_keys = set(settings.keys()) - written_keys
    if new_keys:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append("\n# Added by settings page\n")
        for key in sorted(new_keys):
            lines.append(f"{key}={settings[key]}\n")
    with open(ENV_PATH, "w") as f:
        f.writelines(lines)


def parse_csv_preview(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            rows.append({
                "time": row[0].strip(),
                "import_price": row[1].strip(),
                "export_price": row[2].strip(),
            })
    return rows


class DevHandler(http.server.SimpleHTTPRequestHandler):
    """Handler that serves static files and settings API endpoints."""

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_POST(self):
        if self.path == "/api/settings":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                write_env_file(body)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 400)

        elif self.path == "/api/settings/upload-pricing":
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self._send_json({"ok": False, "error": "Expected multipart form data"}, 400)
                    return
                boundary = content_type.split("boundary=")[1].strip()
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                boundary_bytes = boundary.encode()
                parts = raw.split(b"--" + boundary_bytes)
                file_content = None
                filename = "custom_pricing.csv"
                for part in parts:
                    if b"filename=" in part:
                        header_end = part.index(b"\r\n\r\n")
                        header = part[:header_end].decode("utf-8", errors="replace")
                        for segment in header.split(";"):
                            segment = segment.strip()
                            if segment.startswith("filename="):
                                filename = segment.split("=", 1)[1].strip('"')
                        file_content = part[header_end + 4:]
                        if file_content.endswith(b"\r\n"):
                            file_content = file_content[:-2]
                        break
                if file_content is None:
                    self._send_json({"ok": False, "error": "No file found in upload"}, 400)
                    return
                dest = ROOT / filename
                with open(dest, "wb") as f:
                    f.write(file_content)
                preview = parse_csv_preview(str(dest))
                env_settings = read_env_file()
                env_settings["CUSTOM_PRICING_CSV"] = filename
                write_env_file(env_settings)
                self._send_json({"ok": True, "filename": filename, "preview": preview})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 400)

        elif self.path == "/api/override":
            self._send_json({"ok": True, "override": "AUTO", "expires": None})

        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/api/settings":
            self._send_json(read_env_file())

        elif self.path == "/api/settings/pricing-preview":
            try:
                env = read_env_file()
                csv_file = env.get("CUSTOM_PRICING_CSV", "")
                if csv_file:
                    csv_path = ROOT / csv_file
                    if csv_path.exists():
                        preview = parse_csv_preview(str(csv_path))
                        self._send_json({"ok": True, "preview": preview})
                        return
                self._send_json({"ok": False, "error": "No pricing CSV configured"})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 400)

        else:
            super().do_GET()


def serve(port: int = 8080):
    """Start HTTP server serving the web directory."""
    os.chdir(WEB_DIR)
    server = http.server.HTTPServer(("", port), DevHandler)
    url = f"http://localhost:{port}"
    print(f"\nDashboard ready at {url}")
    print(f"Settings page at {url}/settings.html")
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
