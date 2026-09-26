#!/usr/bin/env python3
"""Generate the anonymous copy of the interactive illustration.

The public page lives in the Jekyll tabs collection and is served from the
site; the anonymous copy is a standalone folder meant to be uploaded next to
the anonymized code, so it carries its own index.html, styles.css and js/.

    python3 scripts/make_anonymous_notebook.py

Source  _tabs/length-independent-state-tracking.html  (+ the folder of the
        same name, which holds styles.css, js/ and assets/)
Target  notebook_iclr/index.html  (+ styles.css and js/ copied over)

Everything the public page says stays, except what names the authors: the
byline, the mail button, the link back to the personal site, and the credit
line of the footer. The assets/ folder of the target is left alone, since the
PCA animations are large and identical in both copies. Every substitution is
checked, so a change in the shape of the public page fails the script instead
of silently producing a page that still carries a name.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_PAGE = ROOT / "_tabs" / "length-independent-state-tracking.html"
PUBLIC_DIR = ROOT / "length-independent-state-tracking"
TARGET_DIR = ROOT / "notebook_iclr"

# Anything that must not survive into the anonymous copy.
IDENTIFYING = [
    "brandoit",
    "uliege",
    "Fyon",
    "Braipson",
    "Clara",
    "De Geeter",
    "Sacr",
    "Ernst",
    "Drion",
    "Montefiore",
    "Li&egrave;ge",
    "Liège",
]


def sub(pattern: str, repl: str, text: str, what: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.S)
    if n != 1:
        sys.exit(f"make_anonymous_notebook: could not find {what} in the public page")
    return out


def anonymize(page: str) -> str:
    # Jekyll front matter: the standalone copy is not built by Jekyll.
    page = sub(r"\A---\n.*?\n---\n", "", page, "the front matter")

    # Keep the copy out of search results if it is ever served from a domain.
    page = sub(
        r'(    <meta name="viewport"[^\n]*\n)',
        r'\1    <meta name="robots" content="noindex, nofollow" />\n',
        page,
        "the viewport meta tag",
    )

    # The top bar: no link home, no mail button.
    page = sub(r'\s*<a class="site-back"[^\n]*\n', "\n", page, "the link back to the site")
    page = sub(r'\s*<a class="nb-btn ghost" href="mailto:[^\n]*\n', "\n", page, "the contact button")

    # The byline.
    page = sub(r'\s*<p class="authors">.*?</p>\n', "\n", page, "the author block")

    # The credit line of the footer.
    page = sub(
        r", by Julien Brandoit.*?University of Li&egrave;ge\)\.",
        ",\n          an anonymous submission under review.",
        page,
        "the footer credit",
    )
    return page


def main() -> None:
    page = anonymize(PUBLIC_PAGE.read_text())

    left = [s for s in IDENTIFYING if s.lower() in page.lower()]
    if left:
        sys.exit("make_anonymous_notebook: the page still names " + ", ".join(left))

    TARGET_DIR.mkdir(exist_ok=True)
    (TARGET_DIR / "index.html").write_text(page)
    shutil.copy2(PUBLIC_DIR / "styles.css", TARGET_DIR / "styles.css")
    shutil.rmtree(TARGET_DIR / "js", ignore_errors=True)
    shutil.copytree(PUBLIC_DIR / "js", TARGET_DIR / "js")

    scripts = len(list((TARGET_DIR / "js").glob("*.js")))
    print(f"notebook_iclr/: index.html, styles.css and {scripts} scripts written")
    if not (TARGET_DIR / "assets" / "pca_projections").is_dir():
        print("warning: notebook_iclr/assets/pca_projections is missing, N12 will not load")


if __name__ == "__main__":
    main()
