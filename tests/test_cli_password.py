"""The optional shared password (--password-file / PRISM_PASSWORD).

The unit tests cover where the password comes from; the integration test
starts a real Panel/Bokeh server with the very options prism passes to
pn.serve, because what matters is that the login is actually enforced for
the page and for the websocket, not that an option was parsed.
"""
import asyncio
import socket
import time

import pytest
import requests

from prism import cli


def test_no_password_means_no_auth_options():
    assert cli.auth_options(None) == {}
    assert cli.auth_options("") == {}


def test_password_becomes_basic_auth_with_a_random_cookie_secret():
    first, second = cli.auth_options("hunter2"), cli.auth_options("hunter2")
    assert first["basic_auth"] == "hunter2"
    # a fresh secret per start: everyone has to log in again after a restart
    assert len(first["cookie_secret"]) >= 32 and first["cookie_secret"] != second["cookie_secret"]


def test_password_comes_from_the_first_line_of_the_file(tmp_path):
    f = tmp_path / "pw"
    f.write_text("s3cret\nignored second line\n")
    assert cli.resolve_password(str(f), environ={}) == "s3cret"


def test_file_wins_over_the_environment(tmp_path):
    f = tmp_path / "pw"
    f.write_text("from-file\n")
    assert cli.resolve_password(str(f), environ={cli.PASSWORD_ENV: "from-env"}) == "from-file"
    assert cli.resolve_password(None, environ={cli.PASSWORD_ENV: "from-env"}) == "from-env"
    assert cli.resolve_password(None, environ={}) is None


@pytest.mark.parametrize("content", ["", "\n"])
def test_empty_password_file_is_an_error_not_an_open_server(tmp_path, content):
    # Silently serving WITHOUT a password because the file was blank would be
    # the worst way to fail.
    f = tmp_path / "pw"
    f.write_text(content)
    with pytest.raises(ValueError, match="empty"):
        cli.resolve_password(str(f), environ={})


def test_missing_password_file_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        cli.resolve_password(str(tmp_path / "nope"), environ={})


def test_parse_args_exits_on_an_unreadable_password_file(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["--password-file", str(tmp_path / "nope")])
    assert "--password-file" in capsys.readouterr().err


def test_parse_args_reads_the_password(tmp_path):
    f = tmp_path / "pw"
    f.write_text("pw1\n")
    assert cli.parse_args(["--password-file", str(f)]).password == "pw1"
    assert cli.parse_args([]).password is None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(**options):
    import panel as pn

    port = _free_port()
    server = pn.serve(
        lambda: pn.pane.Markdown("secret page"), port=port, address="127.0.0.1", show=False,
        threaded=True, **options)
    url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            requests.get(url, timeout=1, allow_redirects=False)
            break
        except requests.ConnectionError:
            time.sleep(0.1)
    return server, url, port


@pytest.fixture
def protected_server():
    server, url, _ = _serve(**cli.auth_options("open-sesame"))
    yield url
    server.stop()


def test_server_enforces_the_login(protected_server):
    url = protected_server

    # not logged in: redirected to the login page, never shown the app
    r = requests.get(url + "/", allow_redirects=False, timeout=5)
    assert r.status_code in (301, 302, 303) and "login" in r.headers["Location"]

    # wrong password (any username is accepted, the password decides)
    s = requests.Session()
    bad = s.post(url + "/login", data={"username": "x", "password": "wrong"}, allow_redirects=False, timeout=5)
    assert "error" in bad.headers.get("Location", "").lower() or bad.status_code in (401, 403)
    assert s.get(url + "/", allow_redirects=False, timeout=5).status_code in (301, 302, 303)

    # right password: the app is served
    s = requests.Session()
    s.post(url + "/login", data={"username": "anyone", "password": "open-sesame"}, allow_redirects=False, timeout=5)
    page = s.get(url + "/", timeout=5)
    assert page.status_code == 200 and "login" not in page.url


def _websocket_handshake(port: int, cookie: str | None = None) -> str:
    from tornado.httpclient import HTTPRequest
    from tornado.websocket import websocket_connect

    headers = {"Origin": f"http://localhost:{port}"}
    if cookie:
        headers["Cookie"] = cookie

    async def connect():
        try:
            conn = await websocket_connect(HTTPRequest(f"ws://127.0.0.1:{port}/ws", headers=headers),
                                           subprotocols=["bokeh", "e30"])
        except Exception as exc:
            return f"rejected {getattr(exc, 'code', type(exc).__name__)}"
        conn.close()
        return "accepted"

    return asyncio.run(connect())


def test_websocket_is_behind_the_login_too():
    # The app's data travels over the websocket, so protecting only the page
    # would not protect anything. The unprotected server is the control: it
    # proves the handshake is rejected because of the login, not because of
    # the test's own origin or token.
    open_server, _, open_port = _serve()
    try:
        assert _websocket_handshake(open_port) == "accepted"
    finally:
        open_server.stop()

    server, url, port = _serve(**cli.auth_options("open-sesame"))
    try:
        assert _websocket_handshake(port) == "rejected 403"
        s = requests.Session()
        s.post(url + "/login", data={"username": "anyone", "password": "open-sesame"}, allow_redirects=False, timeout=5)
        cookie = "; ".join(f"{c.name}={c.value}" for c in s.cookies)
        assert cookie and _websocket_handshake(port, cookie) == "accepted"
    finally:
        server.stop()
