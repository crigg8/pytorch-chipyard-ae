#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


MODULE_PATH = Path(__file__).with_name("prepare-firesim-hwdb.py")
SPEC = importlib.util.spec_from_file_location("prepare_firesim_hwdb_tested", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BundledHwdbTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "bitstreams"
        self.root.mkdir()
        for name in MODULE.BUNDLED_HW_CONFIGS:
            bundle = self.root / name
            bundle.mkdir()
            (bundle / "firesim.tar.gz").write_bytes(b"bitstream")
            (bundle / "driver-bundle.tar.gz").write_bytes(b"driver")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_materializes_all_configs_without_archive_metadata(self) -> None:
        self.assertFalse((self.root / "config_hwdb.yaml.in").exists())
        self.assertFalse((self.root / "SHA256SUMS").exists())

        hwdb = MODULE.load_hwdb(self.root.resolve())
        self.assertEqual(list(hwdb), list(MODULE.BUNDLED_HW_CONFIGS))
        for name, target_config in MODULE.BUNDLED_HW_CONFIGS.items():
            entry = hwdb[name]
            self.assertEqual(
                entry["deploy_quintuplet"]["target_config"], target_config
            )
            self.assertEqual(
                entry["bitstream_tar"], (self.root / name / "firesim.tar.gz").as_uri()
            )
            self.assertEqual(
                entry["driver_tar"],
                (self.root / name / "driver-bundle.tar.gz").as_uri(),
            )

        output = Path(self.temp_dir.name) / "runtime" / "config_hwdb.yaml"
        MODULE.write_atomic(output, hwdb)
        self.assertEqual(yaml.safe_load(output.read_text(encoding="utf-8")), hwdb)

    def test_rejects_a_missing_bundle_file(self) -> None:
        name = next(iter(MODULE.BUNDLED_HW_CONFIGS))
        (self.root / name / "firesim.tar.gz").unlink()
        with self.assertRaises(MODULE.PreparationError):
            MODULE.load_hwdb(self.root.resolve())


if __name__ == "__main__":
    unittest.main()
