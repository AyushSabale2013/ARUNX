import sys
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

import merge_lightcurves
from create_forecast_labels import build_observation_segments
from acquire_data import archive_target, import_portal_script
from validate_preprocessing_run import validate_master


class PreprocessingCoreTests(unittest.TestCase):
    def test_samples_are_binned_to_minute_end_grid(self):
        start = int(pd.Timestamp("2026-01-01T00:00:00Z").value // 1000)
        samples = np.array([start, start + 59_000_000, start + 60_000_000])
        minute, inside = merge_lightcurves._minute_index(samples, start)
        self.assertTrue(inside.all())
        self.assertEqual(minute.tolist(), [0, 0, 1])

    def test_gti_mask_respects_inclusive_interval_endpoints(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        mask = merge_lightcurves.gti_mask(
            times,
            np.array([1.0]),
            np.array([2.0]),
        )
        self.assertEqual(mask.tolist(), [False, True, True, False])

    def test_observation_segments_break_at_time_gaps_and_unusable_rows(self):
        timestamps = pd.Series(pd.to_datetime([
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:02:00Z",
            "2026-01-01T00:04:00Z",
            "2026-01-01T00:05:00Z",
            "2026-01-01T00:06:00Z",
        ], utc=True))
        usable = pd.Series([True, True, True, False, True])
        ids = build_observation_segments(timestamps, usable).tolist()
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[1], ids[2])
        self.assertEqual(ids[3], -1)
        self.assertNotEqual(ids[2], ids[4])

    def test_minute_end_midnight_is_assigned_to_previous_bin_date(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-01-01T23:59:00Z",
                "2026-01-02T00:00:00Z",
            ], utc=True),
            "date": ["2026-01-01", "2026-01-01"],
            "hel1os_demo": [1.0, 2.0],
            "hel1os_demo_gti": [True, True],
            "hel1os_demo_missing": [False, False],
        })
        result = validate_master(frame, "test")
        self.assertEqual(result["date_row_counts"], {"2026-01-01": 2})
        self.assertFalse(any("stored date disagrees" in issue for issue in result["issues"]))
        self.assertTrue(any("daily row counts" in issue for issue in result["issues"]))

    def test_archive_target_rejects_unsupported_instruments_and_paths(self):
        with self.assertRaises(ValueError):
            archive_target("OTHER", "20260101", "file.zip")
        with self.assertRaises(ValueError):
            archive_target("HEL1OS", "20260101", "../file.zip")

    def test_portal_selector_import_reads_urls_without_executing_or_importing_cookie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            selector = root / "selector.py"
            marker = root / "executed.txt"
            manifest.write_text(json.dumps({"schema_version": 1, "entries": []}), encoding="utf-8")
            selector.write_text(
                "url_prefix = 'https://portal.example/data'\n"
                "data_file_paths = ['/HLS_20260101_test.zip']\n"
                f"COOKIE = 'SECRET_COOKIE_SENTINEL'\nopen({str(marker)!r}, 'w').write('ran')\n",
                encoding="utf-8",
            )

            result = import_portal_script(manifest, selector, "HEL1OS", date(2026, 1, 1), date(2026, 1, 1))
            saved = manifest.read_text(encoding="utf-8")

            self.assertEqual(len(result["entries"]), 1)
            self.assertEqual(result["entries"][0]["source_url"], "https://portal.example/data/HLS_20260101_test.zip")
            self.assertNotIn("SECRET_COOKIE_SENTINEL", saved)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
