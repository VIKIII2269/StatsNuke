"""Bundle the website and one snapshot into a single self-contained page (a private
preview: the access flow is simulated in memory, nothing is sent anywhere).

usage: python scripts/build_site_preview.py SNAPSHOT.json OUT.html [--document]
``--document`` wraps it in a full HTML document (for a local browser); without it the
output is a page body for an artifact host that supplies the skeleton.
"""

from __future__ import annotations

import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
FONTS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
    "&family=VT323&display=swap"
)


def main() -> None:
    snap, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = snap.read_text().replace("</", "<\\/")
    body = "\n".join(
        [
            "<title>Stats Nuke</title>",
            f'<link rel="stylesheet" href="{FONTS}">',
            f"<style>\n{(WEB / 'styles.css').read_text()}\n</style>",
            '<div id="app"></div>',
            "<script>window.STATSNUKE_CONFIG = {};\n"
            f"window.STATSNUKE_SNAPSHOT = {data};</script>",
            f"<script>\n{(WEB / 'app.js').read_text()}\n</script>",
        ]
    )
    if "--document" in sys.argv:
        body = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"</head><body>{body}</body></html>"
        )
    out.write_text(body)
    print(f"{out}: {out.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
