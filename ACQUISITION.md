# Data acquisition and archive handling

## Source and authentication

The source is the ISRO PRADAN/ISSDC Aditya-L1 archive. It requires an authenticated portal session and presents selection-specific URLs. The generated downloader snapshots under `downloaders/` are session exports, not stable API clients. Their embedded session values have been removed; the scripts now read `PRADAN_COOKIE` from the environment.

## Reproducible manifest flow

1. Select the required HEL1OS and SoLEXS products and date range in PRADAN and save its generated selector scripts locally.
2. Initialize the acquisition manifest with local archive inventory and checksums:

   ```powershell
   python scripts\pipeline\acquire_data.py --initialize --start 2026-01-01 --end 2026-07-31
   ```

3. Import URL lists without running the portal-generated scripts or importing their cookie constants:

   ```powershell
   python scripts\pipeline\acquire_data.py --portal-hel1os-script <HEL1OS-selector.py> --portal-solexs-script <SoLEXS-selector.py> --start 2026-01-01 --end 2026-07-31
   ```

4. Set an authenticated `PRADAN_COOKIE` in the process environment using a local secret mechanism, then run:

   ```powershell
   python scripts\pipeline\acquire_data.py --download
   ```

The manifest at `output/acquisition_manifest.json` records source URL, archive path, date, size, expected/computed SHA-256, status, and errors. It contains no credentials. Downloads use HTTPS, bounded timeouts, retries with backoff, `.part` files with HTTP range requests for resumption, ZIP CRC validation, and optional expected SHA-256. A valid local archive is skipped. An invalid pre-existing archive is left untouched and reported; it is not silently replaced. Entries without a URL are reported as `blocked_missing_url`.

## Current local evidence

The workspace already has bulk archives, an archive inventory, a prior ZIP integrity report, selective-extraction results, and extracted raw FITS. The current preprocessing config covers 2026-01-01 through 2026-07-31. The multi-day merge has previously produced 207 complete 1,440-row UTC date grids. Archive names and file counts are not equivalent to continuous exposure; use the GTIs and scientific-quality report for instrument coverage.

## Extracting products

`scripts/coverage/extract_archives.py` selectively extracts HEL1OS detector light curves and GTIs and SoLEXS SDD2 light curves and GTIs. ZIP paths are constrained to the raw-data roots, member sizes and CRCs are checked, existing files are verified rather than blindly overwritten, and extraction failures are written to CSV. Re-run with `--start-date`, `--end-date`, and `--report-dir` to scope extraction and keep the report with a preprocessing run.
