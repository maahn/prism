"""Command-line entry point: running `prism` launches the viewer locally."""
from __future__ import annotations

import argparse

import panel as pn

from prism.app import build_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="prism")
    parser.add_argument("--port", type=int, default=0, help="port to serve on (0 = pick automatically)")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = parser.parse_args()
    pn.serve(build_app, port=args.port, show=not args.no_browser, title="PRISM")


if __name__ == "__main__":
    main()
