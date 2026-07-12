#!/usr/bin/env python3
"""Validate the dependency-free PathoSynVLM project website.

The check deliberately uses only the Python standard library so it can run on a
fresh workstation and in GitHub Actions without installing frontend tooling.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


EXPECTED_ORIGIN = "https://atlasanalyticslab.github.io"
EXPECTED_BASE_PATH = "/PathoSynVLM/"
EXPECTED_URL = f"{EXPECTED_ORIGIN}{EXPECTED_BASE_PATH}"
REQUIRED_FILES = {
    "index.html",
    "404.html",
    "robots.txt",
    "sitemap.xml",
    ".nojekyll",
    "static/css/site.css",
    "static/js/site.js",
    "static/images/favicon.svg",
    "static/images/paper_architecture.png",
    "static/images/reported_results.svg",
}
MAX_ASSET_BYTES = 5 * 1024 * 1024


@dataclass
class Document:
    path: Path
    ids: set[str] = field(default_factory=set)
    duplicate_ids: set[str] = field(default_factory=set)
    links: list[tuple[str, str]] = field(default_factory=list)
    images_without_alt: list[str] = field(default_factory=list)
    title_parts: list[str] = field(default_factory=list)
    in_title: bool = False
    descriptions: list[str] = field(default_factory=list)
    canonicals: list[str] = field(default_factory=list)
    viewports: list[str] = field(default_factory=list)
    html_language: str | None = None

    @property
    def title(self) -> str:
        return "".join(self.title_parts).strip()


class SiteHTMLParser(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.document = Document(path=path)

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {name.lower(): value for name, value in attrs}

        element_id = values.get("id")
        if element_id:
            if element_id in self.document.ids:
                self.document.duplicate_ids.add(element_id)
            self.document.ids.add(element_id)

        if tag == "html":
            self.document.html_language = values.get("lang")
        elif tag == "title":
            self.document.in_title = True
        elif tag == "meta":
            name = (values.get("name") or "").lower()
            if name == "description" and values.get("content"):
                self.document.descriptions.append(values["content"] or "")
            if name == "viewport" and values.get("content"):
                self.document.viewports.append(values["content"] or "")
        elif tag == "link":
            rel = (values.get("rel") or "").lower().split()
            href = values.get("href")
            if "canonical" in rel and href:
                self.document.canonicals.append(href)

        for attribute in ("href", "src"):
            value = values.get(attribute)
            if value:
                self.document.links.append((attribute, value.strip()))

        if tag == "img":
            alt = values.get("alt")
            if alt is None or not alt.strip():
                self.document.images_without_alt.append(values.get("src") or "<unknown>")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.document.in_title = False

    def handle_data(self, data: str) -> None:
        if self.document.in_title:
            self.document.title_parts.append(data)


def parse_document(path: Path) -> Document:
    parser = SiteHTMLParser(path)
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser.document


def local_target(
    root: Path, document: Document, url: str
) -> tuple[Path | None, str | None, str | None]:
    """Return (target path, fragment, error) for a local URL."""

    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc:
        return None, None, None

    if not parsed.path and parsed.fragment:
        return document.path, unquote(parsed.fragment), None

    if url == "#":
        return None, None, "placeholder href '#' is not allowed"

    raw_path = unquote(parsed.path)
    if raw_path.startswith("/"):
        if not raw_path.startswith(EXPECTED_BASE_PATH):
            return None, None, f"root-relative URL does not start with {EXPECTED_BASE_PATH}"
        raw_path = raw_path[len(EXPECTED_BASE_PATH) :]
        target = root / raw_path
    else:
        target = document.path.parent / raw_path

    if not raw_path or raw_path.endswith("/"):
        target /= "index.html"

    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        return None, None, "local URL escapes the site root"

    return target, unquote(parsed.fragment) or None, None


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "root", nargs="?", default="site", type=Path, help="static site root"
    )
    args = argument_parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []

    if not root.is_dir():
        print(f"ERROR: site root does not exist: {root}", file=sys.stderr)
        return 1

    for relative_path in sorted(REQUIRED_FILES):
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing required file: {relative_path}")

    html_paths = sorted(root.rglob("*.html"))
    documents: dict[Path, Document] = {}
    for path in html_paths:
        try:
            document = parse_document(path)
        except (OSError, UnicodeError) as error:
            errors.append(f"{path.relative_to(root)}: cannot parse: {error}")
            continue

        documents[path.resolve()] = document
        relative = path.relative_to(root)
        if document.html_language != "en":
            errors.append(f"{relative}: expected <html lang=\"en\">")
        if not document.title:
            errors.append(f"{relative}: missing a non-empty <title>")
        if not document.viewports:
            errors.append(f"{relative}: missing viewport metadata")
        for duplicate_id in sorted(document.duplicate_ids):
            errors.append(f"{relative}: duplicate id '{duplicate_id}'")
        for source in document.images_without_alt:
            errors.append(f"{relative}: image needs non-empty alt text: {source}")

    index_path = (root / "index.html").resolve()
    index = documents.get(index_path)
    if index:
        if len(index.descriptions) != 1 or not index.descriptions[0].strip():
            errors.append("index.html: expected exactly one non-empty meta description")
        if index.canonicals != [EXPECTED_URL]:
            errors.append(
                f"index.html: canonical URL must be exactly {EXPECTED_URL!r}"
            )

    for document in documents.values():
        relative = document.path.relative_to(root)
        for attribute, url in document.links:
            if re.match(r"^(?:mailto|tel|data|javascript):", url, re.IGNORECASE):
                continue
            if url.startswith("http://"):
                errors.append(f"{relative}: insecure external URL: {url}")
                continue

            target, fragment, target_error = local_target(root, document, url)
            if target_error:
                errors.append(f"{relative}: {attribute}={url!r}: {target_error}")
                continue
            if target is None:
                continue
            if not target.is_file():
                errors.append(
                    f"{relative}: {attribute}={url!r} points to missing "
                    f"{target.relative_to(root)}"
                )
                continue
            if fragment:
                target_document = documents.get(target.resolve())
                if target_document and fragment not in target_document.ids:
                    errors.append(
                        f"{relative}: {attribute}={url!r} points to missing id "
                        f"'{fragment}' in {target.relative_to(root)}"
                    )

    for asset in sorted(path for path in root.rglob("*") if path.is_file()):
        size = asset.stat().st_size
        if size > MAX_ASSET_BYTES:
            errors.append(
                f"{asset.relative_to(root)}: {size / (1024 * 1024):.1f} MiB exceeds "
                f"the {MAX_ASSET_BYTES // (1024 * 1024)} MiB per-file budget"
            )

    sitemap_path = root / "sitemap.xml"
    if sitemap_path.is_file():
        try:
            sitemap = ET.parse(sitemap_path)
            namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            locations = [
                (element.text or "").strip()
                for element in sitemap.findall("sm:url/sm:loc", namespace)
            ]
            if EXPECTED_URL not in locations:
                errors.append(f"sitemap.xml: missing {EXPECTED_URL}")
        except ET.ParseError as error:
            errors.append(f"sitemap.xml: invalid XML: {error}")

    robots_path = root / "robots.txt"
    if robots_path.is_file():
        robots = robots_path.read_text(encoding="utf-8")
        expected_sitemap = f"Sitemap: {EXPECTED_URL}sitemap.xml"
        if expected_sitemap not in robots:
            errors.append(f"robots.txt: missing {expected_sitemap!r}")

    if errors:
        print(f"Static-site validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

    file_count = sum(1 for path in root.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(
        "Static-site validation passed: "
        f"{len(documents)} HTML documents, {file_count} files, "
        f"{total_bytes / 1024:.1f} KiB."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
