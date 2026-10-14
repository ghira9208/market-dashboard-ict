"""One-time OAuth setup for the cTrader Open API integration. Run this
once — from the isolated .venv_ctrader environment (see this project's
own README/.gitignore note on why ctrader-open-api can't share the main
venv: it pulls in protobuf 3.20.1, which conflicts with streamlit's own
>=5.26.1 requirement) — to authorize this app against your FxPro/cTrader
account and save the resulting access/refresh tokens locally:

    source .venv_ctrader/bin/activate
    python3 ctrader_auth.py

Opens your browser to cTrader's own consent screen; you log in and click
Allow there, never here.

Scope is "trading" — deliberately widened from this project's original
"accounts" (view-only) scope, now that the edge-pipeline project's own
bot needs real demo-account order execution via cTrader (not just price
data). The resulting broker implementation (see edge-pipeline's
brokers/ctrader.py) explicitly selects and hard-locks onto a DEMO
ctidTraderAccountId — the same "never touch real capital" guarantee
already enforced for eToro — so a "trading"-scoped token, which covers
both live and demo accounts under this cTrader ID, never gets used
against the live one by that code."""
import http.server
import json
import os
import socket
import subprocess
import time
import tomllib
import urllib.parse
import webbrowser

from ctrader_open_api import Auth

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_SECRETS_PATH = os.path.join(_PROJECT_DIR, ".streamlit", "secrets.toml")
_TOKENS_PATH = os.path.join(_PROJECT_DIR, ".cache", "ctrader_tokens.json")
_REDIRECT_URI = "http://localhost:8600/callback"
_PORT = 8600
# The registered redirect URI (localhost:8600) is the dashboard API's own
# port, so the callback can't be received while that service runs — every
# re-auth used to fail with "Address already in use" until someone paused
# it by hand. Paused here instead, and always restored afterward.
_API_JOB = f"gui/{os.getuid()}/com.edgepipeline.api"
_API_PLIST = os.path.expanduser("~/Library/LaunchAgents/com.edgepipeline.api.plist")


def _port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pause_api_if_blocking():
    """Returns True if it stopped the dashboard API (caller must restore)."""
    if not _port_in_use(_PORT):
        return False
    stopped = subprocess.run(["launchctl", "bootout", _API_JOB], capture_output=True).returncode == 0
    if stopped:
        print(f"Paused the dashboard API (it was holding port {_PORT}) — restoring it when done.")
    return stopped


def _restore_api():
    # launchctl bootstrap intermittently fails with a transient "Input/output
    # error" right after a bootout (seen repeatedly on this machine) — a
    # silent single attempt left the dashboard down. Retry until it's
    # actually loaded, and say so loudly if it never is.
    for _ in range(6):
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", _API_PLIST], capture_output=True)
        time.sleep(2)
        if _port_in_use(_PORT):
            print("Dashboard API restored.")
            return
    print(f"WARNING: couldn't restart the dashboard API — run: launchctl bootstrap gui/{os.getuid()} {_API_PLIST}")


def _load_secrets():
    with open(_SECRETS_PATH, "rb") as f:
        return tomllib.load(f)


def main():
    secrets = _load_secrets()
    client_id = secrets["CTRADER_CLIENT_ID"]
    client_secret = secrets["CTRADER_CLIENT_SECRET"]

    auth = Auth(client_id, client_secret, _REDIRECT_URI)
    # product=web matches cTrader's own documented example URL exactly;
    # the SDK's own getAuthUri doesn't add it, so append it ourselves
    # rather than assume it's optional.
    auth_uri = auth.getAuthUri(scope="trading") + "&product=web"

    result = {}

    class _CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            code = qs.get("code", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            if code:
                result["code"] = code
                self.wfile.write(b"<html><body>Authorized -- you can close this tab.</body></html>")
            else:
                self.wfile.write(b"<html><body>No authorization code received.</body></html>")

        def log_message(self, format, *args):
            pass  # quiet -- don't spam stdout with HTTP access logs

    api_paused = _pause_api_if_blocking()
    try:
        # Bind BEFORE opening the browser: a bind failure used to happen
        # after the user had already been sent through a login whose
        # callback had nowhere to land.
        server = http.server.HTTPServer(("localhost", _PORT), _CallbackHandler)
        print("Opening your browser to authorize this app ('trading' scope)...")
        print(auth_uri)
        webbrowser.open(auth_uri)
        print(f"Waiting for the redirect on {_REDIRECT_URI} ...")
        while "code" not in result:
            server.handle_request()
        server.server_close()

        token_response = auth.getToken(result["code"])
        if "accessToken" not in token_response:
            print("Token exchange failed:", token_response)
            return

        os.makedirs(os.path.dirname(_TOKENS_PATH), exist_ok=True)
        with open(_TOKENS_PATH, "w") as f:
            json.dump(token_response, f, indent=2)
        print(f"Saved tokens to {_TOKENS_PATH}")
        print("Access token valid ~30 days; refresh token has no expiry "
              "(ctrader_data.py refreshes automatically before it needs to).")
    finally:
        if api_paused:
            _restore_api()


if __name__ == "__main__":
    main()
