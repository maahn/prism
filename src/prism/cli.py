"""Command-line entry point: running `prism` launches the viewer locally, and
`prism --server` serves independent sessions to many users at once."""
from __future__ import annotations

import argparse
import logging
import os
import secrets

import panel as pn

from prism import server_mode as sm
from prism import settings_store as ss
from prism.app import build_app

SERVER_DEFAULT_PORT = 5006
PASSWORD_ENV = "PRISM_PASSWORD"


def resolve_password(password_file: str | None, environ=os.environ) -> str | None:
    """The shared password, from --password-file (first line) or, failing
    that, the PRISM_PASSWORD environment variable. None = no password.
    Raises ValueError for an unreadable or empty file."""
    if password_file:
        try:
            with open(password_file, encoding="utf-8") as fh:
                password = fh.readline().rstrip("\r\n")
        except OSError as exc:
            raise ValueError(f"cannot read --password-file {password_file}: {exc.strerror}") from exc
        if not password:
            raise ValueError(f"--password-file {password_file} is empty")
        return password
    return environ.get(PASSWORD_ENV) or None


def auth_options(password: str | None) -> dict:
    """pn.serve keyword arguments that put a login page in front of the app
    (and its websocket) when a password is set; empty otherwise.

    Panel's basic_auth takes the password itself rather than a file path --
    a string that happens to name an existing file would be read as a JSON
    user table -- and any username is accepted. The cookie secret is random
    per start, so everyone has to log in again after a restart."""
    if not password:
        return {}
    return {"basic_auth": password, "cookie_secret": secrets.token_hex(32)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="prism")
    parser.add_argument("--port", type=int, default=0,
                        help="port to serve on (0 = pick automatically; 5006 with --server)")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    parser.add_argument("--config", default=os.environ.get("PRISM_CONFIG"), metavar="FILE",
                        help="JSON file pointing sites at shared data directories "
                             '({"site_dirs": {"hyytiala": "/data/hyytiala/cloudnet"}}); '
                             "also PRISM_CONFIG. Disables the cache-size label and 'Clean cache'")
    parser.add_argument("--password-file", default=None, metavar="FILE",
                        help="require a shared password (the first line of FILE; any username) before "
                             f"serving anything; also the {PASSWORD_ENV} environment variable. Sent in "
                             "clear text unless the connection is protected by something else, e.g. an "
                             "SSH tunnel or a TLS-terminating proxy")
    server = parser.add_argument_group(
        "server mode", "serve many users at once, each with an independent session")
    server.add_argument("--server", action="store_true",
                        help="run as a long-lived multi-user server: per-session settings that aren't saved "
                             "to disk, a shared data cache (so 'Clean cache' is disabled), idle sessions "
                             "are released, and no browser is opened")
    server.add_argument("--address", default=None,
                        help="interface to listen on (default with --server: 0.0.0.0, i.e. all interfaces)")
    server.add_argument("--allow-websocket-origin", action="append", default=[], metavar="HOST[:PORT]",
                        help="host[:port] users type into their browser to reach this server, e.g. "
                             "prism.example.org or 192.0.2.7:5006; repeatable. Required for anyone but "
                             "localhost to connect ('*' allows any origin)")
    server.add_argument("--idle-timeout", type=float, default=sm.DEFAULT_IDLE_TIMEOUT_S / 60, metavar="MINUTES",
                        help="release a session's memory after this many minutes without user activity "
                             "(default: %(default)g; 0 = never)")
    args = parser.parse_args(argv)
    args.config_data = None
    if args.config:
        try:
            args.config_data = ss.load_config(args.config)
        except ValueError as exc:
            parser.error(str(exc))
    if args.idle_timeout < 0:
        parser.error("--idle-timeout must be >= 0")
    try:
        args.password = resolve_password(args.password_file)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.server:
        def build_local_session():
            return build_app(config=args.config_data)

        pn.serve(build_local_session, port=args.port, show=not args.no_browser, title="PRISM",
                 **auth_options(args.password))
        return

    # Files written into a shared data directory must stay writable by the
    # other users of its group.
    os.umask(0o002)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    port = args.port or SERVER_DEFAULT_PORT
    if not args.allow_websocket_origin:
        logging.getLogger(__name__).warning(
            "No --allow-websocket-origin given: only http://localhost:%d will work. Pass the host[:port] "
            "your users browse to (e.g. --allow-websocket-origin prism.example.org).", port)

    # A plain function, not functools.partial: Panel only treats real
    # functions as per-session app factories.
    def build_session():
        return build_app(server_mode=True, idle_timeout_s=args.idle_timeout * 60,
                         config=args.config_data)

    logging.getLogger(__name__).info(
        "Password protection %s", "enabled" if args.password else "off (anyone who can reach the port can use PRISM)")
    pn.serve(
        build_session,
        port=port,
        address=args.address or "0.0.0.0",
        websocket_origin=args.allow_websocket_origin or None,
        show=False,
        title="PRISM",
        # A session whose browser never connected (or vanished) is dropped
        # after a minute; connected-but-idle tabs are handled by the app.
        check_unused_sessions_milliseconds=15_000,
        unused_session_lifetime_milliseconds=60_000,
        **auth_options(args.password),
    )


if __name__ == "__main__":
    main()
