"""Inventory and optionally download PRADAN archives from a manifest.

Credentials are read from PRADAN_COOKIE in the process environment. Never put
session cookies or passwords in the manifest. The manifest does not guess URLs:
copy exact archive URLs from an authenticated PRADAN export when adding entries.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


BASE = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = BASE / "output" / "acquisition_manifest.json"
DEFAULT_ARCHIVE_ROOT = BASE / "downloaders" / "pradan1.issdc.gov.in" / "al1" / "protected" / "downloadData"
ARCHIVE_PATTERNS = {
    "HEL1OS": re.compile(r"HLS_(20\d{6})_.*\.zip$", re.I),
    "SoLEXS": re.compile(r"AL1_SLX_L1_(20\d{6})_.*\.zip$", re.I),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC failure in ZIP member: {bad_member}")


def in_date_range(value: str, start: date | None, end: date | None) -> bool:
    parsed = date.fromisoformat(value)
    return (start is None or parsed >= start) and (end is None or parsed <= end)


def archive_target(instrument: str, day: str, filename: str) -> Path:
    if instrument not in ARCHIVE_PATTERNS:
        raise ValueError(f"Unsupported instrument: {instrument}")
    if Path(filename).name != filename:
        raise ValueError(f"Archive filename must not contain a path: {filename}")
    if not re.fullmatch(r"20\d{6}", day):
        raise ValueError(f"Invalid archive date: {day}")
    return DEFAULT_ARCHIVE_ROOT / instrument.lower() / "level1" / day[:4] / day[4:6] / day[6:8] / filename


def target_for_row(row: dict[str, Any]) -> Path:
    relative = str(row.get("relative_path", "")).strip()
    if not relative:
        return archive_target(row["instrument"], row["date"], row["filename"])
    target = (BASE / relative).resolve()
    root = DEFAULT_ARCHIVE_ROOT.resolve()
    if not target.is_relative_to(root):
        raise ValueError("Manifest relative_path must stay inside the PRADAN archive directory")
    if target.name != row["filename"]:
        raise ValueError("Manifest relative_path filename does not match filename field")
    return target


def initialize_manifest(
    manifest_path: Path,
    archive_root: Path,
    start: date | None,
    end: date | None,
) -> dict[str, Any]:
    existing: dict[str, Any] = {}
    if manifest_path.exists() and manifest_path.stat().st_size:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = {
        (row.get("instrument"), row.get("date"), row.get("filename")): row
        for row in existing.get("entries", [])
    }
    integrity_cache: dict[tuple[str, str], int] = {}
    cached_report = BASE / "output" / "archive_integrity_report.csv"
    if cached_report.exists():
        import csv
        with cached_report.open("r", newline="", encoding="utf-8-sig") as stream:
            for cached in csv.DictReader(stream):
                if cached.get("status") == "OK":
                    try:
                        integrity_cache[(cached["instrument"], cached["filename"])] = int(cached["size_bytes"])
                    except (KeyError, TypeError, ValueError):
                        continue
    rows: list[dict[str, Any]] = []
    present_keys: set[tuple[str, str, str]] = set()

    for instrument, pattern in ARCHIVE_PATTERNS.items():
        source = archive_root / instrument.lower() / "level1"
        if not source.exists():
            continue
        for path in sorted(source.rglob("*.zip")):
            match = pattern.fullmatch(path.name)
            if not match:
                continue
            day = match.group(1)
            iso_day = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
            if not in_date_range(iso_day, start, end):
                continue
            key = (instrument, day, path.name)
            present_keys.add(key)
            row = previous.get(key, {})
            try:
                if integrity_cache.get((instrument, path.name)) != path.stat().st_size:
                    validate_zip(path)
                    integrity_source = "fresh_full_crc_check"
                else:
                    integrity_source = "existing_full_crc_report_and_size_match"
                digest = sha256_file(path)
                status, error = "local_verified", ""
            except Exception as exc:
                digest = ""
                status, error = "local_invalid", str(exc)
            rows.append({
                "instrument": instrument,
                "date": day,
                "filename": path.name,
                "source_url": row.get("source_url", ""),
                "relative_path": str(path.relative_to(BASE)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "expected_sha256": row.get("expected_sha256", ""),
                "sha256": digest,
                "integrity_source": integrity_source if status == "local_verified" else "",
                "status": status,
                "error": error,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            })

    # Preserve manifest-planned remote entries that do not yet exist locally.
    rows.extend(
        row for key, row in previous.items()
        if key not in present_keys and str(row.get("source_url", "")).strip()
    )

    result = {
        "schema_version": 1,
        "source": "ISRO PRADAN / ISSDC Aditya-L1 archive",
        "requested_date_range": {
            "start_inclusive": start.isoformat() if start else None,
            "end_inclusive": end.isoformat() if end else None,
        },
        "credential_policy": "PRADAN_COOKIE environment variable; no secret values stored",
        "entries": rows,
    }
    atomic_json(manifest_path, result)
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_portal_script(manifest_path: Path, script_path: Path, instrument: str, start: date | None, end: date | None) -> dict[str, Any]:
    """Import the selected archive URL list from a PRADAN-generated Python file.

    AST parsing reads only literal URL-list assignments; it never executes the
    downloaded script and never copies its cookie/header constants.
    """
    if instrument not in ARCHIVE_PATTERNS:
        raise ValueError(f"Unsupported instrument: {instrument}")
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"data_file_paths", "url_prefix"}:
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        continue
    paths = values.get("data_file_paths")
    if not isinstance(paths, list):
        raise ValueError(f"No literal data_file_paths list found in {script_path}")
    prefix = values.get("url_prefix", "https://pradan1.issdc.gov.in")
    if not isinstance(prefix, str) or not prefix.startswith("https://"):
        raise ValueError("PRADAN url_prefix must be HTTPS")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() and manifest_path.stat().st_size else {
        "schema_version": 1,
        "source": "ISRO PRADAN / ISSDC Aditya-L1 archive",
        "credential_policy": "PRADAN_COOKIE environment variable; no secret values stored",
        "entries": [],
    }
    entries = manifest.setdefault("entries", [])
    by_key = {(row.get("instrument"), row.get("date"), row.get("filename")): row for row in entries}
    imported = 0
    from urllib.parse import urljoin
    for value in paths:
        if not isinstance(value, str):
            continue
        url = urljoin(prefix.rstrip("/") + "/", value.lstrip("/"))
        filename = Path(urlparse(url).path).name
        match = ARCHIVE_PATTERNS[instrument].fullmatch(filename)
        if not match:
            continue
        day = match.group(1)
        iso_day = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        if not in_date_range(iso_day, start, end):
            continue
        key = (instrument, day, filename)
        row = by_key.get(key)
        if row is None:
            row = {
                "instrument": instrument,
                "date": day,
                "filename": filename,
                "source_url": url,
                "relative_path": "",
                "size_bytes": 0,
                "expected_sha256": "",
                "sha256": "",
                "status": "planned",
                "error": "",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            entries.append(row)
            by_key[key] = row
        else:
            row["source_url"] = url
            if row.get("status") not in {"local_verified", "already_present_verified", "downloaded_verified"}:
                row["status"] = "planned"
        imported += 1
    manifest["portal_selection_scripts"] = manifest.get("portal_selection_scripts", [])
    manifest["portal_selection_scripts"].append({
        "instrument": instrument,
        "filename": script_path.name,
        "sha256": sha256_file(script_path),
        "urls_imported": imported,
        "note": "URL list imported from literal AST assignments; script was not executed and cookies were not imported.",
    })
    if start or end:
        manifest["requested_date_range"] = {
            "start_inclusive": start.isoformat() if start else None,
            "end_inclusive": end.isoformat() if end else None,
        }
    atomic_json(manifest_path, manifest)
    return manifest


def download_entry(
    row: dict[str, Any],
    cookie: str,
    timeout: int,
    retries: int,
    backoff: float,
    chunk_size: int,
) -> tuple[str, str, int, str]:
    url = str(row.get("source_url", "")).strip()
    if not url:
        return "blocked_missing_url", "No source_url configured; URL was not guessed", 0, ""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        return "invalid_url", "Only absolute HTTPS source URLs are accepted", 0, ""

    target = target_for_row(row)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            validate_zip(target)
            digest = sha256_file(target)
            expected = str(row.get("expected_sha256", "")).strip().lower()
            if not expected or digest.lower() == expected:
                return "already_present_verified", "", target.stat().st_size, digest
            return "existing_checksum_mismatch", "Existing archive does not match expected_sha256; left untouched", target.stat().st_size, digest
        except Exception as exc:
            return "existing_invalid", f"Existing archive failed validation and was left untouched: {exc}", target.stat().st_size, ""
    part = target.with_name(target.name + ".part")
    expected = str(row.get("expected_sha256", "")).strip().lower()
    headers = {"Cookie": cookie} if cookie else {}
    last_error = "download failed"

    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        request_headers = dict(headers)
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(
                url,
                headers=request_headers,
                stream=True,
                timeout=(min(20, timeout), timeout),
                allow_redirects=True,
            ) as response:
                if response.status_code in (401, 403):
                    return "auth_required", f"HTTP {response.status_code}; refresh PRADAN session cookie", 0, ""
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                with part.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())

            validate_zip(part)
            digest = sha256_file(part)
            if expected and digest.lower() != expected:
                raise ValueError("Downloaded SHA-256 does not match expected_sha256")
            size = part.stat().st_size
            part.replace(target)
            return "downloaded_verified", "", size, digest
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
    return "failed", last_error, part.stat().st_size if part.exists() else 0, ""


def run_downloads(manifest_path: Path, timeout: int, retries: int, backoff: float, chunk_size: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("entries"), list):
        raise ValueError("Unsupported or malformed acquisition manifest")
    cookie = os.environ.get("PRADAN_COOKIE", "").strip()
    rows = []
    for row in manifest["entries"]:
        status, error, size, digest = download_entry(row, cookie, timeout, retries, backoff, chunk_size)
        row.update(status=status, error=error, size_bytes=size, sha256=digest,
                   updated_utc=datetime.now(timezone.utc).isoformat())
        rows.append(row)
    manifest["entries"] = rows
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--initialize", action="store_true", help="Inventory and hash local archives in the selected date range")
    parser.add_argument("--download", action="store_true", help="Download manifest entries with source_url values; credentials come from PRADAN_COOKIE")
    parser.add_argument("--portal-hel1os-script", type=Path, help="PRADAN-generated HEL1OS selector script; URL list only is imported")
    parser.add_argument("--portal-solexs-script", type=Path, help="PRADAN-generated SoLEXS selector script; URL list only is imported")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff", type=float, default=5)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    args = parser.parse_args()
    if args.start and args.end and args.start > args.end:
        parser.error("--start must be on or before --end")
    if args.initialize:
        manifest = initialize_manifest(args.manifest, args.archive_root, args.start, args.end)
        print(f"Inventoried {len(manifest['entries'])} archives: {args.manifest}")
    if args.portal_hel1os_script:
        manifest = import_portal_script(args.manifest, args.portal_hel1os_script, "HEL1OS", args.start, args.end)
        print(f"Imported portal HEL1OS list: {len(manifest['entries'])} total manifest entries")
    if args.portal_solexs_script:
        manifest = import_portal_script(args.manifest, args.portal_solexs_script, "SoLEXS", args.start, args.end)
        print(f"Imported portal SoLEXS list: {len(manifest['entries'])} total manifest entries")
    if args.download:
        manifest = run_downloads(args.manifest, args.timeout, args.retries, args.backoff, args.chunk_size)
        counts: dict[str, int] = {}
        for row in manifest["entries"]:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        print(f"Updated {len(manifest['entries'])} archive entries: {counts}")
    if not (args.initialize or args.portal_hel1os_script or args.portal_solexs_script or args.download):
        parser.error("choose --initialize or --download")


if __name__ == "__main__":
    main()
