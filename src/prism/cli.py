"""Command-line entry point: running `prism` launches the viewer locally, and
`prism --server` serves independent sessions to many users at once."""
from __future__ import annotations

import argparse
import logging

import panel as pn

from prism import server_mode as sm
from prism.app import build_app

SERVER_DEFAULT_PORT = 5006


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="prism")
    parser.add_argument("--port", type=int, default=0,
                        help="port to serve on (0 = pick automatically; 5006 with --server)")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
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
    if args.idle_timeout < 0:
        parser.error("--idle-timeout must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.server:
        pn.serve(build_app, port=args.port, show=not args.no_browser, title="PRISM")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    port = args.port or SERVER_DEFAULT_PORT
    if not args.allow_websocket_origin:
        logging.getLogger(__name__).warning(
            "No --allow-websocket-origin given: only http://localhost:%d will work. Pass the host[:port] "
            "your users browse to (e.g. --allow-websocket-origin prism.example.org).", port)

    # A plain function, not functools.partial: Panel only treats real
    # functions as per-session app factories.
    def build_session():
        return build_app(server_mode=True, idle_timeout_s=args.idle_timeout * 60)

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
    )


if __name__ == "__main__":
    main()
