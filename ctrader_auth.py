"""One-time OAuth setup for the cTrader Open API integration. Run this
once — from the isolated .venv_ctrader environment (see this project's
own README/.gitignore note on why ctrader-open-api can't share the main
venv: it pulls in protobuf 3.20.1, which conflicts with streamlit's own
>=5.26.1 requirement) — to authorize this app against your FxPro/cTrader
account and save the resulting access/refresh tokens locally:

    source .venv_ctrader/bin/activate
    python3 ctrader_auth.py

Opens your browser to cTrader's own consent screen; you log in and click
Allow there, never here. Scope is deliberately "accounts" (view-only),
not "trading" — the resulting access token is then structurally
incapable of placing an order, regardless of what any calling code does
with it. Never widen this to "trading" for this project — the whole
point of this integration is price data, not execution.

See ctrader_data.py for the actual client that uses the saved tokens."""
import http.server
import json
import os
import tomllib
import urllib.parse
import webbrowser

from ctrader_open_api import Auth

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_SECRETS_PATH = os.path.join(_PROJECT_DIR, ".streamlit", "secrets.toml")
_TOKENS_PATH = os.path.join(_PROJECT_DIR, ".cache", "ctrader_tokens.json")
_REDIRECT_URI = "http://localhost:8600/callback"
_PORT = 8600


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
    auth_uri = auth.getAuthUri(scope="accounts") + "&product=web"

    print("Opening your browser to authorize this app (view-only 'accounts' scope)...")
    print(auth_uri)
    webbrowser.open(auth_uri)

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

    server = http.server.HTTPServer(("localhost", _PORT), _CallbackHandler)
    print(f"Waiting for the redirect on {_REDIRECT_URI} ...")
    while "code" not in result:
        server.handle_request()

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


if __name__ == "__main__":
    main()
