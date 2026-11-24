"""Generate the versioned end-to-end preprocessing technical manual."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak,
    Table, LongTable, TableStyle, KeepTogether, NextPageTemplate,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "preproc_20261124_v1"
RUN = ROOT / "output/preprocessing_runs" / RUN_ID
MASTER_DIR = ROOT / "output/master_datasets" / RUN_ID
PDF = ROOT / "docs/ADITYA_L1_COMPLETE_DATA_PREPROCESSING_DOCUMENTATION.pdf"
STATUS = json.loads((RUN / "reports/preprocessing_report.json").read_text(encoding="utf-8"))
RUN_MANIFEST = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
FEATURE_SCHEMA = json.loads((RUN / "ml_ready/feature_schema.json").read_text(encoding="utf-8"))
MASTER_MANIFEST = json.loads((MASTER_DIR / "manifest.json").read_text(encoding="utf-8"))
SUMMARY = pd.read_csv(RUN / "ml_ready/dataset_summary.csv")

NAVY = colors.HexColor("#10243A")
BLUE = colors.HexColor("#1D6A96")
CYAN = colors.HexColor("#16A6A1")
PALE = colors.HexColor("#EAF2F7")
INK = colors.HexColor("#263544")
MUTED = colors.HexColor("#5B6B78")
AMBER = colors.HexColor("#FFF2D9")
RED = colors.HexColor("#9A3F36")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitleX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=27, leading=33, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12))
styles.add(ParagraphStyle(name="CoverDeckX", parent=styles["Normal"], fontSize=13, leading=19, textColor=MUTED, spaceAfter=18))
styles.add(ParagraphStyle(name="H1X", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=NAVY, spaceBefore=10, spaceAfter=10, keepWithNext=True))
styles.add(ParagraphStyle(name="H2X", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=BLUE, spaceBefore=9, spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=9.2, leading=13.2, textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallX", parent=styles["BodyText"], fontSize=7.6, leading=10, textColor=INK, spaceAfter=3))
styles.add(ParagraphStyle(name="TinyX", parent=styles["BodyText"], fontSize=6.8, leading=8.3, textColor=INK))
styles.add(ParagraphStyle(name="CalloutX", parent=styles["BodyText"], fontSize=9, leading=13, textColor=NAVY, backColor=PALE, borderColor=CYAN, borderWidth=0.8, borderPadding=8, spaceBefore=5, spaceAfter=9))
styles.add(ParagraphStyle(name="WarnX", parent=styles["BodyText"], fontSize=9, leading=13, textColor=RED, backColor=AMBER, borderColor=colors.HexColor("#D9A441"), borderWidth=0.8, borderPadding=8, spaceBefore=5, spaceAfter=9))
styles.add(ParagraphStyle(name="TableHeadX", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.2, leading=8.5, textColor=colors.white))
styles.add(ParagraphStyle(name="TOCX", parent=styles["BodyText"], fontSize=10, leading=15, textColor=NAVY))


def P(text: str, style="BodyX") -> Paragraph:
    return Paragraph(text, styles[style])


def para_list(items, style="BodyX"):
    return [P("• &nbsp;" + x, style) for x in items]


def make_table(rows, widths, small=False, repeatRows=1):
    table = LongTable(rows, colWidths=widths, repeatRows=repeatRows, hAlign="LEFT", splitByRow=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CFD9E0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
    ]))
    return table


def feature_description(name: str) -> str:
    if name in ("hel1os_available", "solexs_available"):
        return "Int8 indicator: at least one non-null HEL1OS signal" if name == "hel1os_available" else "Int8 indicator: SoLEXS SDD2 count is non-null"
    if name in ("time_sin", "time_cos"):
        return "Sine/cosine encoding of UTC hour + minute/60, period 24 h; timestamp-derived known-at-observation-time feature."
    if name in ("day_year_sin", "day_year_cos"):
        return "Sine/cosine encoding of UTC day-of-year, period 365.25 d; timestamp-derived known-at-observation-time feature."
    match = re.match(r"^(.*)_roll(\d+)_(mean|std|count|min|max)$", name)
    if match:
        channel, w, stat = match.groups()
        return f"{stat} of log1p({channel}) over the current and preceding {w} one-minute samples, within one continuous usable segment; min_periods=2 for std, otherwise 1."
    match = re.match(r"^(.*)_diff_(\d+)m$", name)
    if match:
        channel, lag = match.groups()
        return f"log1p({channel}) at t minus its value {lag} minutes earlier in the same usable segment; no future sample used."
    if name.endswith("_stat_err_log1p"):
        return f"log1p of HEL1OS statistical error for {name[:-len('_stat_err_log1p')]} (source error values are passed through log1p)."
    if name.endswith("_stat_err_missing"):
        return f"Int8 missingness flag for HEL1OS statistical error of {name[:-len('_stat_err_missing')]}."
    if name.endswith("_log1p"):
        return f"log1p of the nonnegative merged count/rate signal {name[:-len('_log1p')]}; missing remains NaN."
    if name.endswith("_missing"):
        return f"Int8 copy of source missingness indicator for {name[:-len('_missing')]}."
    if name.endswith("_gti"):
        return f"Int8 copy of the source Good Time Interval (GTI) coverage flag for {name[:-len('_gti')]}."
    return "Feature name from the frozen run feature schema; see scripts/pipeline/build_ml_dataset.py for the exact transformation."


class ManualDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=20*mm, bottomMargin=17*mm, title="Aditya-L1 End-to-End Data Preprocessing Technical Documentation", author="ARUN-X Data Pipeline")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="body", frames=[frame], onPage=self.decorate)])

    def decorate(self, canvas, doc):
        if doc.page > 1:
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#D4DEE5")); canvas.setLineWidth(.5)
            canvas.line(18*mm, A4[1]-14*mm, A4[0]-18*mm, A4[1]-14*mm)
            canvas.setFont("Helvetica-Bold", 7); canvas.setFillColor(NAVY)
            canvas.drawString(18*mm, A4[1]-11*mm, "ARUN-X  /  ADITYA-L1 DATA PREPROCESSING")
            canvas.setFont("Helvetica", 7); canvas.setFillColor(MUTED)
            canvas.drawRightString(A4[0]-18*mm, 10*mm, f"{RUN_ID}  •  {doc.page}")
            canvas.restoreState()


def main():
    PDF.parent.mkdir(parents=True, exist_ok=True)
    story = []
    story += [Spacer(1, 26*mm), P("ADITYA-L1 / HEL1OS + SoLEXS", "H2X"),
              P("Complete Data Preprocessing Pipeline", "CoverTitleX"),
              P("Implementation-grounded technical documentation from archive acquisition and FITS parsing through labels, 592 engineered features, chronological task datasets, and a versioned canonical master.", "CoverDeckX"),
              Spacer(1, 8*mm)]
    facts = [[P("RUN", "TableHeadX"), P("ROWS", "TableHeadX"), P("FEATURES", "TableHeadX"), P("TASK FAMILIES", "TableHeadX")],
             [P(RUN_ID), P("298,080", "H2X"), P("592", "H2X"), P("6", "H2X")]]
    t = Table(facts, colWidths=[42*mm, 38*mm, 38*mm, 54*mm], rowHeights=[9*mm, 17*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("BACKGROUND",(0,1),(-1,1),PALE),("GRID",(0,0),(-1,-1),.5,colors.white),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(0,0),(-1,-1),"CENTER")]))
    story += [t, Spacer(1, 10*mm), P("Documentation version 1.0  |  24 November 2026  |  Status: PASS_WITH_REVIEW", "BodyX"),
              P("Scope: what this code actually does for preproc_20261124_v1. It distinguishes artifacts inherited from earlier runs and labels untested behavior as untested.", "CalloutX"),
              P("Canonical master: output/master_datasets/preproc_20261124_v1/canonical_master_features_targets.parquet (298,080 × 705). The source run files are preserved unchanged.", "BodyX"), PageBreak()]

    story += [P("Contents", "H1X")]
    toc = ["1. Executive overview and run facts", "2. Instrument and source context", "3. Pipeline architecture and exact processing sequence", "4. Archive acquisition and provenance", "5. ZIP extraction and FITS parsing", "6. Time normalization, GTIs, merging and quality flags", "7. GOES event parsing and labels", "8. Forecast target construction", "9. Feature engineering and full 592-feature dictionary", "10. Task tables, row policies and chronological splits", "11. Validation, reproducibility and data lineage", "12. Findings, limitations and recommended implementation work", "Appendix A. Generated file inventory", "Appendix B. Schema and data-type notes", "Appendix C. Sources and code references"]
    story += [P(x, "TOCX") for x in toc] + [Spacer(1, 5*mm), P("Navigation note: use the PDF viewer’s page thumbnails/search. The feature dictionary lists every approved model input by exact name and its derivation rule.", "SmallX"), PageBreak()]

    story += [P("1. Executive overview and run facts", "H1X"),
      P("This run converts locally verified Aditya-L1 HEL1OS and SoLEXS daily archive products plus a GOES X-ray event catalogue into aligned UTC one-minute samples, flare labels, segment-aware future-onset targets, 592 past-only features, and six task families split by calendar time. The run is marked PASS_WITH_REVIEW; this is a data-pipeline status, not evidence of predictive generalization.")]
    story += [P("Run facts", "H2X")]
    overview = [
      ("Run ID / timestamp", RUN_ID + " / 2026-11-24 12:49:56 UTC"),
      ("Requested interval", "2026-01-01 through 2026-07-31 inclusive (212 calendar dates)"),
      ("Merged coverage", "207 dates; 298,080 rows; 1,440 rows per represented date"),
      ("Grid", "1-minute UTC; right-edge label. First row 2026-01-01 00:01 UTC; last row 2026-08-01 00:00 UTC represents the final minute of July 31."),
      ("Features / tasks", "592 ordered features; classification, nowcasting, peak_detection, forecast_15m, forecast_30m, forecast_60m"),
      ("Source run status", "PASS_WITH_REVIEW; 23 manifest-listed output files (18 split tables plus 5 schema/summary artifacts)."),
      ("Master artifact", "298,080 rows × 705 columns: 592 features, six targets, source columns, timestamp/date, and two identifiers."),
    ]
    story += [make_table([[P("Item","TableHeadX"),P("Run value","TableHeadX")]] + [[P(a,"SmallX"),P(b,"SmallX")] for a,b in overview], [42*mm,130*mm])]
    story += [Spacer(1,5*mm), P("Pipeline at a glance", "H2X")]
    steps = ["1  Authenticated portal selection / local archive inventory", "↓", "2  ZIP validation + selective FITS extraction", "↓", "3  HEL1OS and SoLEXS FITS + GTI reading", "↓", "4  UTC conversion → 1-min bins → instrument merge", "↓", "5  GOES parse → active/class/peak labels", "↓", "6  segment-aware 15/30/60-min onset targets", "↓", "7  past-only 592-feature matrix", "↓", "8  time-ordered task tables + validation manifest"]
    flow = Table([[P(x,"SmallX")] for x in steps], colWidths=[102*mm], hAlign="LEFT")
    flow.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),PALE),("BOX",(0,0),(-1,-1),.7,CYAN),("INNERGRID",(0,0),(-1,-1),.3,colors.white),("LEFTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story += [flow, PageBreak()]

    story += [P("2. Instrument and source context", "H1X"),
      P("SoLEXS is a soft X-ray spectrometer and HEL1OS a hard X-ray spectrometer on Aditya-L1. This repository consumes level-2-like FITS light curves and auxiliary Good Time Interval products supplied through the PRADAN archive workflow; GOES supplies event timing and class labels. Official mission material describes SoLEXS as covering 2–22 keV and HEL1OS as covering hard X-rays with detector products. See Appendix C for primary mission/data references.")]
    story += [P("Inputs used by the implementation", "H2X"),
      make_table([[P("Input","TableHeadX"),P("Read fields / semantics","TableHeadX"),P("Pipeline role","TableHeadX")],
        [P("HEL1OS lightcurve FITS", "SmallX"),P("Each band HDU must expose MJD, CTR, STAT_ERR; exactly five band HDUs per detector segment are expected.","SmallX"),P("Four detectors × five channels; signal + statistical error.","SmallX")],
        [P("HEL1OS GTI FITS", "SmallX"),P("GTI extension has TSTART/TSTOP; finite intervals with STOP > START retained.","SmallX"),P("Mask out-of-GTI counts/errors; preserve per-channel coverage.","SmallX")],
        [P("SoLEXS lightcurve FITS", "SmallX"),P("RATE table with TIME and COUNTS; TIME interpreted as seconds from Unix epoch (MJDREF=40587).","SmallX"),P("One SDD2 count signal consumed in the merge.","SmallX")],
        [P("SoLEXS GTI FITS", "SmallX"),P("START/STOP intervals; timestamps tested against inclusive endpoints.","SmallX"),P("Mask samples outside exposure.","SmallX")],
        [P("GOES catalogue", "SmallX"),P("Rows parse date and start/peak/end clocks plus class string; event times normalize to UTC and midnight rollover is handled.","SmallX"),P("M/X event labels and forecast-onset targets.","SmallX")],
      ], [35*mm,77*mm,60*mm])]
    story += [P("The merger assigns stable band indices 1–5 in FITS-HDU order and channel names such as hel1os_cdte1_band1. It does not persist calibrated energy boundaries as feature metadata in the 592-column schema. Downstream users must not infer exact band-energy edges from the band index alone; add and validate the header-derived boundaries before physical band interpretation.", "WarnX"), PageBreak()]

    story += [P("3. Pipeline architecture and processing sequence", "H1X")]
    stage_rows = [
      ("Stage 0 — Configuration", "config/preprocessing.json copied into the run; requested dates, local roots, 1-minute grid, both-instrument row policy, four rolling windows, three difference lags, M/X targets, split boundaries, retry parameters."),
      ("Stage 1 — Archive inventory", "Find local HEL1OS and SoLEXS ZIPs and verify ZIP structure; attach per-archive status and hashes in run-local acquisition manifest."),
      ("Stage 2 — Selective extraction", "Extract chosen FITS lightcurve and GTI members to instrument folders, with extraction inventory/status reports."),
      ("Stage 3 — GOES parsing", "Read event list, parse event clocks, reconcile rollover, normalize timestamps, retain catalogue records and flag invalid chronology."),
      ("Stage 4 — Merge", "Read all detector lightcurves and GTIs, normalize times, aggregate samples onto daily UTC minute grid and concatenate represented days."),
      ("Stage 5 — Interval labels", "Mark each completed minute overlapping an M or X event; classify M=1/X=2, preserve the highest class in overlaps, mark peak minute."),
      ("Stage 6 — Forecast targets", "Find next M/X start; emit onset-within-horizon binary labels only where future coverage remains in the same usable segment."),
      ("Stage 7 — Feature matrix", "Build approved 592 inputs; reset historical windows at gaps and unusable periods; save feature schema."),
      ("Stage 8 — Split & validate", "Assign sample by represented UTC date to train/validation/test; purge forecast horizon rows at train/validation boundaries; validate schemas, counts and run hashes."),
    ]
    story += [make_table([[P("Stage","TableHeadX"),P("Implementation behavior","TableHeadX")]] + [[P(a,"SmallX"),P(b,"SmallX")] for a,b in stage_rows], [39*mm,133*mm])]
    story += [Spacer(1,4*mm),P("Important run-specific execution caveat", "H2X"), P("The report records 633 local_verified archives and 3,824 extraction records (3,816 extracted, 8 already_exists). This invocation was run with extraction skipped: its validation report reused/copies a prior raw FITS audit (8,708 FITS files, 370,885,639 samples; 1,905,151 outside GTI masked) rather than demonstrating that remote download and all extraction steps were freshly executed in this invocation. The root acquisition manifest is empty, and the in-run manifest gives no source URLs or expected remote checksums. Treat local verification and inherited QA as evidence with that provenance limitation.","WarnX"), PageBreak()]

    story += [P("4. Archive acquisition and provenance", "H1X"),
      P("The acquisition utility supports local archive inventory and optional download from a manifest. It intentionally does not guess authenticated portal URLs: URL lists can be imported from a portal-generated script using literal AST parsing, avoiding execution of the portal script. Manifest entries include instrument, date, filename, status, local path, file size/hash, source URL, and error/status details when available. ZIP validation and SHA-256 are local integrity checks; they do not establish server provenance without a URL and expected remote digest.")]
    story += [P("What this run proves", "H2X")]
    story += para_list(["633 archives are reported local_verified in the run-local inventory.","The configured retry policy is 90-second request timeout, up to five retries, five-second backoff, 1 MiB chunks; the code supports resumable/chunked HTTP behavior where invoked.","No source URLs or expected remote checksums are recorded for this run. Authenticated remote download was not tested.","Download logs are absent; for absent dates we cannot conclude whether the remote archive was unavailable, not requested/selected, or a download attempt failed.","`--skip-extract` means the first run did not freshly execute the extraction step; prior extraction and raw QA reports were reused/copied."])
    story += [P("Missing requested dates", "H2X")]
    missing = [("2026-02-17", "0 HEL1OS / 0 SoLEXS local archives"),("2026-04-16", "0 / 0"),("2026-05-22", "0 / 0"),("2026-06-04", "0 / 1 SoLEXS"),("2026-06-09", "2 HEL1OS / 0 SoLEXS")]
    story += [make_table([[P("UTC date","TableHeadX"),P("Local archive inventory","TableHeadX"),P("Result","TableHeadX")]]+[[P(d,"SmallX"),P(a,"SmallX"),P("No merged date","SmallX")] for d,a in missing], [40*mm,75*mm,57*mm]),
              Spacer(1,3*mm),P("Consequence: no observations or forecast labels are fabricated for these five dates. The ML generator drops rows without usable signal; the missing calendar gaps break temporal segments, so rolling features and forecast labels do not cross the gaps. Temporal coverage is still incomplete, so event-rate estimates and model evaluation can be biased if archive loss correlates with instrument operations, solar activity, or event reporting. Do not impute or interpolate observational values.","WarnX"), PageBreak()]

    story += [P("5. ZIP extraction and FITS parsing", "H1X"),
      P("The archive extractor inventories ZIP members and selectively extracts the science lightcurve and auxiliary GTI members to a per-instrument raw layout. ZIP CRC/structure verification is distinct from semantic FITS validation. Extraction reports distinguish newly extracted from already-existing files; existing files should be hash-checked before reuse.")]
    story += [P("HEL1OS path", "H2X"),P("For each detector file, the merger locates matching `aux/gti<detector>.fits`; it searches binary-table HDUs for each band table with MJD, CTR and STAT_ERR. There must be five band HDUs. Samples with invalid MJD are removed; GTIs are read as TSTART/TSTOP, rejecting nonfinite or nonpositive intervals. MJD is converted to Unix microseconds. Samples outside GTI retain their time slot but their signal and errors become NaN.")]
    story += [P("SoLEXS path", "H2X"),P("The merger pairs each compressed lightcurve with its `.gti.gz` sidecar, loads a RATE table with TIME and COUNTS, filters nonfinite time values, and masks counts outside the matching GTIs. TIME is interpreted as seconds from the Unix epoch using MJDREF=40587. SoLEXS contributes the SDD2 count channel to this run.")]
    story += [P("Run-specific handling", "H2X"),P("Input timestamp arrays are stable-sorted and duplicate timestamps are rejected. Each UTC day is binned into 1,440 minute slots using integer microsecond offsets from midnight. For HEL1OS, per-minute signal is the mean of valid counts; per-minute error is sqrt(sum(error²))/n over valid errors. A minute with no valid sample for that channel is NaN. A channel’s GTI flag is true if any valid GTI sample falls in that minute. The resulting date grid is labelled at the right edge (00:01 … next day 00:00).")]
    story += [P("Scientific boundary", "H2X"),P("These are aggregated count/rate quantities, not an absolute calibrated flux product. The two instruments have different response and units; no cross-calibration, background subtraction, detector-response inversion, or hard/soft ratio is performed by this preprocessing code. Values should be treated as pipeline signal features, not directly comparable physical fluxes.","WarnX"),PageBreak()]

    story += [P("6. Time normalization, GTIs, merging and quality", "H1X"),
      P("UTC is the common time basis. HEL1OS MJD values are mapped to Unix microseconds; SoLEXS TIME seconds are mapped to the same scale. Daily bins are indexed relative to UTC midnight and the labels placed one minute after each bin start. The date column denotes the represented UTC date, while the timestamp at midnight is the right edge of the preceding date’s final interval.")]
    story += [P("Missingness and availability", "H2X"),P("The merged table retains signal columns, HEL1OS statistical errors, per-signal GTI indicators, per-signal missing indicators and the SoLEXS SDD2 equivalents. The ML row policy is `both_instruments`: a row is eligible when at least one HEL1OS signal and the SoLEXS signal are both usable. Within an eligible row individual channel signals may still be missing; channel-level flags and NaNs carry this information. No signal interpolation is performed.")]
    story += [P("Known caveats", "H2X")]
    story += para_list(["An archive present for a day does not ensure continuous instrument coverage; GTI masks and missingness indicators must be retained.","The merged data has 105 fields and 298,080 rows; labeled and forecast-labeled outputs contain 108 and 111 fields respectively.","The source date range contains five absent calendar days; observed date count is 207 of 212.","The final timestamp 2026-08-01 00:00 UTC is expected: it labels the last one-minute bin belonging to July 31.","A 1-minute average suppresses sub-minute peak shape; exact peak timing can be coarsened to the minute when targets and signals are compared."])
    story += [PageBreak()]

    story += [P("7. GOES event parsing and labels", "H1X"),P("GOES event clock fields are parsed to timestamps; midnight rollover is handled by the parser. The event interval must satisfy start ≤ peak ≤ end. Invalid event order is rejected from event labeling. Only M and X classes feed the current classification, active-event, peak, and forecast-onset labels; A/B/C events are treated as background for these targets.")]
    story += [P("Label rules", "H2X"),make_table([
      [P("Target","TableHeadX"),P("Value / definition","TableHeadX")],
      [P("flare_active","SmallX"),P("1 when an M/X event overlaps the completed sample interval (bin_start, bin_end]; otherwise 0.","SmallX")],
      [P("flare_class","SmallX"),P("0 = no overlapping M/X; 1 = M; 2 = X. For overlapping event intervals, retain the higher class.","SmallX")],
      [P("flare_peak","SmallX"),P("1 on the minute-end row containing the event peak. A peak exactly on a minute boundary belongs to the next completed minute.","SmallX")],
      [P("flare_15m / 30m / 60m","SmallX"),P("1 if the next M/X event start occurs in (t, t+horizon]; 0 if no start is due within horizon and the future is fully observed; -1 unknown for currently active rows or incomplete future usable coverage.","SmallX")],
    ],[42*mm,130*mm])]
    story += [Spacer(1,4*mm),P("Malformed GOES entry", "H2X"),P("One row at line 1642 is recorded as `3-Jul-2026 23:50 12:03 00:05 M2.0`. The clock ordering is invalid after midnight rollover because the nominal peak parses as 12:03, preceding the event’s 23:50 start, while the end 00:05 can be rolled into the next date. It could reflect a typo intended as 00:03, but that correction is not evidenced by the catalogue and must not be invented. Excluding this malformed event is appropriate under the chronology invariant; it prevents an invalid label interval. Any true event represented by that line is absent from labels; nearby minutes may be labeled background incorrectly and the event is unavailable as a positive training/evaluation case. Resolve only from an authoritative catalogue revision.","WarnX"),
      P("Targets overlap only in the interval-label layer when events overlap; flare_class uses highest class. The run audit found no overlapping M/X intervals in its catalogue. Forecast onset targets refer to event starts, not peak times or thresholds inferred from the instrument signals.","BodyX"),PageBreak()]

    story += [P("8. Forecast target construction", "H1X"),
      P("For every minute-end sample t, the target builder locates the first GOES M/X event whose start is strictly later than t. A positive horizon label requires 0 < event_start − t ≤ H minutes. Event starts exactly at t are excluded by the strict future lookup; onset at the horizon boundary is included.")]
    story += [P("Observed future coverage is required", "H2X"),
      P("A forecast label is only considered observed if the row at t+H exists at the exact expected minute, both endpoints remain in the same segment, and that segment is usable. A segment resets at a timestamp gap or an unusable observation. Current M/X-active rows and rows without complete same-segment future coverage are marked -1. For task tables, -1 rows are removed.")]
    story += [P("Boundary protection", "H2X"),P("During dataset assembly, train samples whose target horizon reaches or crosses June 1 are purged. Validation samples whose target horizon reaches or crosses July 1 are purged. Test is left as the terminal held-out period, where right-edge incomplete horizons are already unknown and dropped. This avoids target windows crossing the chronological boundary, but adjacent samples for the same underlying flare can still be present across split boundaries when event lead times or active intervals overlap; event-level grouping is not implemented in the split builder.")]
    target_dist = {"15 min": {"unknown":18393,"negative":277387,"positive":2300},"30 min":{"unknown":18862,"negative":274779,"positive":4439},"60 min":{"unknown":19773,"negative":270067,"positive":8240}}
    rows = [[P("Horizon","TableHeadX"),P("Positive","TableHeadX"),P("Negative","TableHeadX"),P("Unknown excluded","TableHeadX"),P("Positive rate among known","TableHeadX")]]
    for h,d in target_dist.items(): rows.append([P(h,"SmallX"),P(f"{d['positive']:,}","SmallX"),P(f"{d['negative']:,}","SmallX"),P(f"{d['unknown']:,}","SmallX"),P(f"{d['positive']/(d['positive']+d['negative']):.3%}","SmallX")])
    story += [P("Forecast target distribution over all 298,080 rows", "H2X"),make_table(rows,[34*mm,31*mm,33*mm,38*mm,36*mm]),
      Spacer(1,4*mm),P("Negative labels are not simply “no event exists”: they require complete usable observation through the horizon. Dropping unknown rows changes coverage and can induce selection bias if missingness relates to activity or instrument duty cycle.","CalloutX"),PageBreak()]

    story += [P("9. Feature engineering", "H1X"),P("The frozen schema is `ml_ready/feature_schema.json`; exact ordering is used by the generated task datasets. The 592 inputs consist of signal transforms, quality/missingness indicators, rolling summaries, temporal differences, availability indicators, and deterministic UTC calendar encodings. There is no target or GOES event field in the approved input list.")]
    # Count feature families from exact names
    features = FEATURE_SCHEMA["features"]
    counts = {"availability":0,"signal log1p":0,"channel quality flags":0,"error log1p / flag":0,"rolling statistics":0,"temporal differences":0,"UTC calendar encodings":0,"other":0}
    for f in features:
        if f in ("hel1os_available","solexs_available"): counts["availability"]+=1
        elif f in ("time_sin","time_cos","day_year_sin","day_year_cos"): counts["UTC calendar encodings"]+=1
        elif re.match(r".*_roll\d+_",f): counts["rolling statistics"]+=1
        elif re.match(r".*_diff_\d+m$",f): counts["temporal differences"]+=1
        elif f.endswith("_stat_err_log1p") or f.endswith("_stat_err_missing"): counts["error log1p / flag"]+=1
        elif f.endswith("_missing") or f.endswith("_gti"): counts["channel quality flags"]+=1
        elif f.endswith("_log1p"): counts["signal log1p"]+=1
        else: counts["other"]+=1
    story += [make_table([[P("Feature family","TableHeadX"),P("Count","TableHeadX"),P("Definition","TableHeadX")]] + [
      [P(k,"SmallX"),P(str(v),"SmallX"),P({"availability":"Int8 sensor availability indicators.","signal log1p":"log1p of merged sensor signal.","channel quality flags":"Copied missingness and GTI flags as int8.","error log1p / flag":"HEL1OS statistical error log1p plus error-missing flag.","rolling statistics":"Mean, sample standard deviation, count, min, max at 5/15/30/60 min for each signal.","temporal differences":"Within-segment log1p signal differences at 1/5/15 min.","UTC calendar encodings":"Sin/cos hour-of-day and day-of-year features.","other":"No unmatched features; should be zero."}[k],"SmallX")] for k,v in counts.items()], [47*mm,18*mm,107*mm])]
    story += [Spacer(1,3*mm),P("Continuous-segment rule", "H2X"),P("Feature windows use current and historical observations only (pandas right-aligned rolling default); they reset at any timestamp gap and after rows with no usable instrument data. Window values are log1p-transformed signals. Rolling windows have min_periods=1 except standard deviation (2). Differences compare current log1p signal with the same channel’s value at the specified prior lag. No scaling/standardization is fit in preprocessing; any training-time scaler must be fitted on training only. Missing signal values remain NaN for XGBoost’s native missing-value handling.")]
    story += [P("Potential implementation gap", "H2X"),P("The code uses np.log1p without an explicit nonnegative/finite-domain assertion on each source signal. Unexpected negative count/rate values in (-1, 0) can yield finite transforms; values ≤ −1 can produce NaN or −∞. Add per-channel range checks, explicit handling policy, and a test asserting finite transformed values wherever source values are expected to be valid. Also validate that `*_missing` agrees with signal NaN and that a GTI-false value cannot be used as a valid signal.","WarnX"),PageBreak()]

    story += [P("Feature dictionary — all 592 approved inputs", "H1X"),P("The table below is generated from the exact ordered feature schema and the formulas in build_ml_dataset.py. Column names and ordering match the current run. The “meaning” column describes each individual input; repeated formulas differ by source channel/window/lag.","BodyX")]
    dictrows = [[P("#","TableHeadX"),P("Feature name","TableHeadX"),P("Meaning / computation","TableHeadX")]]
    for i,name in enumerate(features,1):
        dictrows.append([P(str(i),"TinyX"),P(name.replace("_","_<wbr/>"),"TinyX"),P(feature_description(name),"TinyX")])
    story += [make_table(dictrows,[9*mm,57*mm,106*mm])]
    story += [PageBreak()]

    story += [P("10. Task tables and chronological splits", "H1X"),P("The ML builder writes 18 Parquet files, one for each of six tasks × train/validation/test. Each task file contains timestamp, a normalized `target` column, and the same 592 ordered features. It excludes rows with no usable signal; the three forecast tasks additionally exclude target=-1. The five supported task families plus? In this run there are six: three event-state tasks and three distinct forecast horizons.")]
    story += [make_table([[P("Task file prefix","TableHeadX"),P("Source target","TableHeadX"),P("Target interpretation","TableHeadX")],
      [P("classification","SmallX"),P("flare_class","SmallX"),P("0 no M/X; 1 M; 2 X","SmallX")],
      [P("nowcasting","SmallX"),P("flare_active","SmallX"),P("0/1 active M/X at current minute","SmallX")],
      [P("peak_detection","SmallX"),P("flare_peak","SmallX"),P("0/1 peak minute","SmallX")],
      [P("forecast_15m","SmallX"),P("flare_15m","SmallX"),P("binary future M/X onset ≤15 min; unknown removed","SmallX")],
      [P("forecast_30m","SmallX"),P("flare_30m","SmallX"),P("binary future M/X onset ≤30 min; unknown removed","SmallX")],
      [P("forecast_60m","SmallX"),P("flare_60m","SmallX"),P("binary future M/X onset ≤60 min; unknown removed","SmallX")],
    ],[43*mm,42*mm,87*mm])]
    story += [P("Chronological boundaries", "H2X"),P("The represented sample date is timestamp minus one minute. Training covers represented dates before 2026-06-01; validation covers June 1 through June 30; test begins July 1. The state-task row counts are 183,280 train / 36,896 validation / 41,047 test. Forecast tasks have fewer rows from unusable observations, unknown horizons, and boundary purging.")]
    # Summary inventory by task split
    splitrows = [[P("Task","TableHeadX"),P("Split","TableHeadX"),P("Rows","TableHeadX"),P("Target distribution","TableHeadX")]]
    for _,r in SUMMARY.iterrows():
        splitrows.append([P(str(r.task),"TinyX"),P(str(r.split),"TinyX"),P(f"{int(r.rows):,}","TinyX"),P(str(r.target_distribution),"TinyX")])
    story += [make_table(splitrows,[38*mm,25*mm,25*mm,84*mm]),PageBreak()]

    story += [P("11. Validation, reproducibility and lineage", "H1X"),P("The preprocessing status report records output validation, source/config hashes, and SHA-256 hashes for all 23 output files. The run manifest names the processing stages and reports file sizes/hashes. The baseline training run links to this dataset run separately; older workspace model files with 424 features are historical artifacts and are not a continuation of this 592-feature schema.")]
    story += [P("Lineage identifiers", "H2X"),P(f"Master SHA-256: {MASTER_MANIFEST['master_sha256']}<br/>Source forecast-labeled table SHA-256: {MASTER_MANIFEST['source_forecast_table_sha256']}<br/>Feature schema SHA-256: {MASTER_MANIFEST['feature_schema_sha256']}<br/>Source run manifest SHA-256: {MASTER_MANIFEST['source_run_manifest_sha256']}<br/>Rows / columns / features: {MASTER_MANIFEST['rows']:,} / {MASTER_MANIFEST['columns']} / {MASTER_MANIFEST['feature_count']}")]
    story += [P("Rebuild", "H2X"),P("From the repository root, using the project virtual environment: `venv\\Scripts\\python.exe scripts\\pipeline\\build_canonical_master.py`. The script reads the immutable run-local forecast-labeled table and frozen feature schema, refuses to overwrite a populated version directory, verifies exact feature names/order, adds stable observation IDs and run IDs, and emits schema and hash manifests. To make a different version, update the explicit RUN_ID and write to a new version path; never replace existing artifacts.")]
    story += [P("Validation boundaries", "H2X"),P("Manifest hashes prove byte-level identity of local artifacts; they do not prove scientific correctness or source authenticity. The inherited raw audit predates this run. The report’s PASS_WITH_REVIEW label does not imply remote-download, training, instrument calibration, or operational forecasting readiness.","WarnX"),PageBreak()]

    story += [P("12. Findings, limitations and recommended work", "H1X"),P("The dataset is usable for exploratory model development under explicit limitations. A robust operational claim would be premature because several unresolved coverage, event, acquisition, and evaluation issues remain.")]
    limitations = [
      ("Five missing days", "207 of 212 requested dates represented; cause cannot be distinguished without portal download records and authenticated retrieval."),
      ("Malformed GOES event", "One M2.0 event excluded because event chronology is invalid; validate against an authoritative revised catalogue."),
      ("Sparse X test cases", "One X-class flare in July test period; X-class performance is not statistically meaningful."),
      ("One test period", "Only one temporal holdout (July 2026); no evidence of generalization across solar cycles, regimes, or later periods."),
      ("Event-level overlap", "Boundary horizon purge is present, but no event-grouped split is enforced/verified; nearby flare samples may cross splits."),
      ("Remote acquisition untested", "No authenticated end-to-end download/log provenance; exact absence reason for dates unknown."),
      ("Energy calibration context", "Band index→physical energy edges are not embedded in the ML feature schema; no response calibration/cross-instrument normalization."),
      ("Potential transform-domain issue", "No explicit negative-input/log1p domain policy or source/flag consistency check documented in feature builder."),
    ]
    story += [make_table([[P("Issue","TableHeadX"),P("Effect","TableHeadX")]]+[[P(a,"SmallX"),P(b,"SmallX")] for a,b in limitations],[44*mm,128*mm])]
    story += [P("Recommended implementation order", "H2X")]
    recs = ["Add a reproducible authenticated portal acquisition job that records exact URLs, retrieval timestamps, HTTP result codes, remote checksums, local hashes, retries, and complete logs; rerun the five dates and classify unavailability vs retrieval failure.","Obtain an authoritative corrected GOES catalogue row or retain explicit excluded-event metadata and mark affected evaluation intervals for review; never hand-correct event clocks.","Add parse/merge/feature invariants: FITS unit and energy-bound extraction, nonnegative signal checks, GTI/missing consistency, source sample cadence checks, and round-trip feature parity checks.","Add event-aware embargo/group splitting around each flare and horizon; retain strict chronological test ordering and ensure events/horizon windows cannot straddle boundaries.","Gather more independent X events and additional test periods before any X-class or generalization claim; quantify uncertainty at event level, not only minute-row level.","Record a full dataset lineage manifest for each master, include exact schema/dtypes, feature family counts, row-level IDs, and immutable source hashes; use a new run ID for any changed transformation.","Evaluate calibration and operational utility with event-based detection, lead time, false alarms per day, PR-AUC and threshold policy, in addition to row-wise metrics."]
    story += para_list(recs, "SmallX")
    story += [P("Readiness decision", "H2X"),P("All six families are READY_WITH_LIMITATIONS for exploratory training per DATASET_READINESS_REPORT.md. None should be presented as operationally validated. X-class test performance is not interpretable with only one X test event, and one test month cannot establish stable generalization.","WarnX"),PageBreak()]

    story += [P("Appendix A. Generated file inventory", "H1X"),P("The run manifest lists 23 outputs. The table lists exact relative path, byte size and SHA-256. It includes the three merged/label master tables, schema/summary CSVs, and all 18 task-split tables.","BodyX")]
    invrows = [[P("Relative path","TableHeadX"),P("Bytes","TableHeadX"),P("SHA-256","TableHeadX")]]
    for item in RUN_MANIFEST["output_files"]:
        invrows.append([P(item["path"],"TinyX"),P(f"{item['size_bytes']:,}","TinyX"),P(item["sha256"],"TinyX")])
    story += [make_table(invrows,[64*mm,23*mm,85*mm]),PageBreak()]

    story += [P("Appendix B. Schema and data-type notes", "H1X"),P("The 105-column merged table carries date/timestamp, instrument signals, HEL1OS statistical errors, GTI indicators, and missing indicators. The 108-column labeled table adds flare_active, flare_class, flare_peak. The 111-column forecast-labeled table adds flare_15m, flare_30m, flare_60m. The model tables contain timestamp and target followed by 592 features. The canonical master preserves the 111 source fields, all 592 feature columns, plus observation_id and source_run_id = 705 columns.")]
    story += [P("Canonical master metadata", "H2X"),make_table([
      [P("Column class","TableHeadX"),P("Fields","TableHeadX"),P("Storage / note","TableHeadX")],
      [P("Identifiers","SmallX"),P("observation_id, source_run_id","SmallX"),P("Stable UTC-minute ID and `preproc_20261124_v1` value.","SmallX")],
      [P("Time metadata","SmallX"),P("date, timestamp","SmallX"),P("Date string + timezone-aware UTC timestamp. Timestamp labels right edge of represented minute.","SmallX")],
      [P("Targets","SmallX"),P("flare_active, flare_class, flare_peak, flare_15m, flare_30m, flare_60m","SmallX"),P("Integer labels: state labels int64 source; forecast labels int8; unknown forecast=-1.","SmallX")],
      [P("Raw observations","SmallX"),P("HEL1OS/SoLEXS signals, errors and source flags","SmallX"),P("Signals/errors float64; source flags bool. Source fields colliding with feature flags use `raw__` prefix and mapping is recorded in schema JSON.","SmallX")],
      [P("Feature inputs","SmallX"),P("592 names from frozen schema","SmallX"),P("528 float64 and 64 int8, matching source feature schema.","SmallX")],
      [P("Storage","SmallX"),P("canonical_master_features_targets.parquet","SmallX"),P("Parquet with Zstandard compression; schema and hashes in adjacent JSON files.","SmallX")],
    ],[35*mm,72*mm,65*mm])]
    story += [Spacer(1,4*mm),P("Artifacts beside the master", "H2X"),P("`canonical_master_schema.json` lists all 705 fields, exact 592 feature names, six targets, metadata/source columns, timestamp convention, compression and raw collision renames. `manifest.json` records master/schema hashes, source run and input hashes, row and column counts, feature count, time coverage, and an explicit note that the original outputs were left unchanged.","BodyX"),PageBreak()]

    story += [P("Appendix C. Sources and code references", "H1X"),P("Primary mission and data references", "H2X"),
      P('<link href="https://www.isro.gov.in/Aditya_L1.html" color="#1D6A96">ISRO — Aditya-L1 mission and payload overview</link>',"BodyX"),
      P('<link href="https://pradan.issdc.gov.in/al1/faq.xhtml" color="#1D6A96">PRADAN — Aditya-L1 data portal FAQ</link>',"BodyX"),
      P('<link href="https://pradan.issdc.gov.in/al1/index.xhtml" color="#1D6A96">PRADAN — Aditya-L1 data portal mission page</link>',"BodyX"),
      P('<link href="https://www.ursc.gov.in/hel1os.jsp" color="#1D6A96">URSC — HEL1OS instrument page</link>',"BodyX"),
      P('<link href="https://www.ursc.gov.in/SoLEXS.jsp" color="#1D6A96">URSC — SoLEXS instrument page</link>',"BodyX"),
      P("Official portal payload specifications are useful context; this implementation report states the columns and transformations actually found in this repository. Refer to current instrument calibration documentation for science-grade interpretation.","SmallX"),
      P("Implementation files inspected", "H2X")]
    code_refs = ["config/preprocessing.json — run interval, grid, row policy, features, targets and split boundaries.","scripts/pipeline/acquire_data.py — local inventory, manifest, optional authenticated archive retrieval and integrity hashes.","scripts/coverage/extract_archives.py — ZIP inventory and selective FITS extraction.","scripts/pipeline/parse_goes.py — event clock parsing and rollover handling.","scripts/pipeline/merge_lightcurves.py — FITS/GTI reading, time conversion, UTC binning and merge.","scripts/pipeline/label_flares.py — active/class/peak target generation.","scripts/pipeline/create_forecast_labels.py — segment-aware horizon labels.","scripts/pipeline/build_ml_dataset.py — feature transformations, eligibility, task outputs and chronological splits.","scripts/validation/validate_preprocessing_run.py — output validation and manifest.","scripts/pipeline/build_canonical_master.py — additive master builder included with this document.","DATA_PREPROCESSING_STATUS.md, DATASET_READINESS_REPORT.md, BASELINE_MODEL_RESULTS.md — run, audit and exploratory training context."]
    story += para_list(code_refs,"SmallX")
    story += [Spacer(1,5*mm),P(f"Run manifest SHA-256: {MASTER_MANIFEST['source_run_manifest_sha256']}<br/>Canonical master SHA-256: {MASTER_MANIFEST['master_sha256']}<br/>Feature schema SHA-256: {MASTER_MANIFEST['feature_schema_sha256']}<br/>Generated: 2026-11-24 (Asia/Kolkata).", "SmallX"),
              Spacer(1,8*mm),P("End of technical documentation", "H2X")]

    doc = ManualDoc(PDF)
    doc.build(story)
    print(PDF)
    print(f"pages generated (post-build): {doc.page}")


if __name__ == "__main__":
    main()
