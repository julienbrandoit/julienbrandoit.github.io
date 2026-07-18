#!/usr/bin/env python3
"""Rewrite local `assets/` resource paths in Markdown for a repo that is both an
Obsidian vault and a Jekyll site.

Two directions, both idempotent and fence-aware (never touches fenced code
blocks, external URLs, mailto:, in-page anchors, or Liquid `{{ ... }}`):

  --to-relative   Normalize every local assets reference to a path relative to
                  the Markdown file's own location, e.g. from `_posts/x.md` an
                  `assets/foo.png` or `/assets/foo.png` becomes `../assets/foo.png`.
                  This is the form that works in Obsidian and in GitHub's preview.

  --to-site       Rewrite the file-relative form back to a site-absolute path for
                  Jekyll, e.g. `../assets/foo.png` becomes `/assets/foo.png`.
                  Run this on a throwaway build copy only, never on the vault.

Only links whose target resolves under `assets/` are ever rewritten; every other
link is left exactly as it is.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Matches the target of a Markdown link or image: the `(...)` after `](`.
LINK_RE = re.compile(r"\]\(([^)]+)\)")

# Directories never scanned (vendored assets, VCS, build output, caches).
SKIP_DIRS = {".git", "_site", ".jekyll-cache", ".sass-cache", "assets", "node_modules"}

# A real fenced-code delimiter: a line that is only a run of 3+ ` or ~ plus an
# info string that contains no more fence characters. This deliberately excludes
# inline uses like ```code``` written on a single line mid-sentence, whose trailing
# backticks would otherwise be misread as opening a block.
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})([^`~]*)$")


def core_assets_path(target: str) -> str | None:
    """Return the `assets/...` tail of a local resource target, or None if the
    target is not a local assets resource (external, anchor, Liquid, other)."""
    t = target.strip()
    lowered = t.lower()
    if lowered.startswith(("http://", "https://", "mailto:", "ftp://", "//")):
        return None
    if t.startswith("#") or t.startswith("{{") or t.startswith("{%"):
        return None
    core = t
    while core.startswith("../"):
        core = core[3:]
    if core.startswith("./"):
        core = core[2:]
    if core.startswith("/"):
        core = core[1:]
    if core.startswith("assets/"):
        return core
    return None


def rewrite_target(target: str, mode: str, depth: int) -> str:
    core = core_assets_path(target)
    if core is None:
        return target
    if mode == "to-relative":
        prefix = "../" * depth if depth > 0 else ""
        return prefix + core
    # to-site
    return "/" + core


def rewrite_text(text: str, mode: str, depth: int) -> tuple[str, int]:
    out_lines = []
    in_fence = False
    fence_marker = ""
    changes = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        m = FENCE_RE.match(stripped)
        if in_fence:
            # Only a matching fence character closes the block; everything else,
            # including link-like text, is code content and is left untouched.
            if m and m.group(1)[0] == fence_marker:
                in_fence = False
                fence_marker = ""
            out_lines.append(line)
            continue
        if m:
            in_fence = True
            fence_marker = m.group(1)[0]
            out_lines.append(line)
            continue

        def repl(match: re.Match) -> str:
            nonlocal changes
            original = match.group(1)
            new = rewrite_target(original, mode, depth)
            if new != original:
                changes += 1
            return f"]({new})"

        out_lines.append(LINK_RE.sub(repl, line))
    return "".join(out_lines), changes


def file_depth(path: str, root: str) -> int:
    rel_dir = os.path.relpath(os.path.dirname(os.path.abspath(path)), os.path.abspath(root))
    if rel_dir in (".", ""):
        return 0
    return len(rel_dir.split(os.sep))


def iter_markdown(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".md"):
                yield os.path.join(dirpath, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="repository root to scan for Markdown files")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--to-relative", action="store_const", const="to-relative", dest="mode")
    mode.add_argument("--to-site", action="store_const", const="to-site", dest="mode")
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = ap.parse_args()

    total_files = 0
    total_changes = 0
    for path in iter_markdown(args.root):
        with open(path, "r", encoding="utf-8") as fh:
            original = fh.read()
        depth = file_depth(path, args.root)
        new, changes = rewrite_text(original, args.mode, depth)
        if changes and new != original:
            total_files += 1
            total_changes += changes
            rel = os.path.relpath(path, args.root)
            print(f"  {rel}: {changes} path(s)")
            if not args.dry_run:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new)
    verb = "would change" if args.dry_run else "changed"
    print(f"{args.mode}: {verb} {total_changes} path(s) across {total_files} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
