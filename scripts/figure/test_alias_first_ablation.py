from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import generate_plot_inputs as plot_inputs


class TestAliasFirstAblation(unittest.TestCase):
    def test_workload_name_parses_alias_first_tags(self):
        model, tags, core, tokens = plot_inputs.parse_workload_name(
            "gpt-neo-gemmini-sdpa-256tok-alias-first-off-4core"
        )

        self.assertEqual(model, "gpt-neo")
        self.assertEqual(tags, ("gemmini", "sdpa", "alias", "first", "off"))
        self.assertEqual(core, 4)
        self.assertEqual(tokens, 256)

    def test_legacy_cnn_name_parses_core_before_alias_tags(self):
        model, tags, core, tokens = plot_inputs.parse_workload_name(
            "resnet50-gemmini-4core-alias-first-off"
        )

        self.assertEqual(model, "resnet50")
        self.assertEqual(tags, ("gemmini", "alias", "first", "off"))
        self.assertEqual(core, 4)
        self.assertIsNone(tokens)

    def test_csv_uses_off_over_on_cycles_as_speedup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_csv_dir = plot_inputs.CSV_DIR
            plot_inputs.CSV_DIR = Path(tmpdir)
            try:
                runs = [
                    self._run(
                        "opt-gemmini-sdpa-256tok-alias-first-off-4core", 200.0
                    ),
                    self._run(
                        "opt-gemmini-sdpa-256tok-alias-first-on-4core", 100.0
                    ),
                    self._run("opt-rocket-gemmini-sdpa-256tok-4core", 50.0),
                ]
                path = plot_inputs.write_alias_first_ablation_csv(runs)
            finally:
                plot_inputs.CSV_DIR = original_csv_dir

            with path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        opt = next(row for row in rows if row["model"] == "opt")
        self.assertEqual(opt["off_cycles"], "200")
        self.assertEqual(opt["on_cycles"], "100")
        self.assertEqual(opt["speedup"], "2.000000")

    def test_csv_compares_cnn_off_with_canonical_gemmini_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_csv_dir = plot_inputs.CSV_DIR
            plot_inputs.CSV_DIR = Path(tmpdir)
            try:
                runs = [
                    self._run("resnet50-gemmini-4core-alias-first-off", 150.0),
                    self._run("resnet50-gemmini-4core", 100.0),
                    self._run("resnet50-gemmini-im2col-4core", 50.0),
                ]
                path = plot_inputs.write_alias_first_ablation_csv(runs)
            finally:
                plot_inputs.CSV_DIR = original_csv_dir

            with path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        resnet = next(row for row in rows if row["model"] == "resnet50")
        self.assertEqual(resnet["off_cycles"], "150")
        self.assertEqual(resnet["on_cycles"], "100")
        self.assertEqual(resnet["speedup"], "1.500000")

    def test_alias_runs_do_not_replace_canonical_sdpa_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_csv_dir = plot_inputs.CSV_DIR
            plot_inputs.CSV_DIR = Path(tmpdir)
            try:
                runs = [
                    self._run("opt-rocket-gemmini-sdpa-256tok-4core", 50.0),
                    self._run(
                        "opt-gemmini-sdpa-256tok-alias-first-off-4core", 200.0
                    ),
                    self._run(
                        "opt-gemmini-sdpa-256tok-alias-first-on-4core", 100.0
                    ),
                ]
                path = plot_inputs.write_sdpa_csv(runs)
            finally:
                plot_inputs.CSV_DIR = original_csv_dir

            with path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        baseline = next(row for row in rows if row["model"] == "opt")
        self.assertEqual(baseline["total_kernel_cycles_avg"], "50")

    def test_rvv_diagnostic_run_does_not_replace_canonical_cnn_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_csv_dir = plot_inputs.CSV_DIR
            plot_inputs.CSV_DIR = Path(tmpdir)
            try:
                runs = [
                    self._run("mobilenetv2-rvv-2core", 100.0),
                    self._run("mobilenetv2-rvv-nohazard-2core", 200.0),
                ]
                path = plot_inputs.write_cnn_result_csv(runs)
            finally:
                plot_inputs.CSV_DIR = original_csv_dir

            with path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        mobile = next(
            row
            for row in rows
            if row["variant"] == "baseline"
            and row["device"] == "Saturn"
            and row["model"] == "MobileNet"
        )
        self.assertEqual(mobile["2 Saturn"], "100")

    def test_alias_run_does_not_supply_attention_baseline(self):
        alias_only = [
            self._run("opt-gemmini-sdpa-256tok-alias-first-off-4core", 200.0)
        ]
        self.assertIsNone(
            plot_inputs.attention_run(alias_only, "opt", 256, "sdpa", host="rocket")
        )

        canonical = self._run("opt-rocket-gemmini-sdpa-256tok-4core", 50.0)
        selected = plot_inputs.attention_run(
            alias_only + [canonical], "opt", 256, "sdpa", host="rocket"
        )
        self.assertIs(selected, canonical)

    @staticmethod
    def _run(workload: str, cycles: float) -> plot_inputs.WorkloadRun:
        model, tags, core, tokens = plot_inputs.parse_workload_name(workload)
        path = Path("/tmp") / workload / "model.log"
        return plot_inputs.WorkloadRun(
            workload=workload,
            result_dir=path.parent,
            model_log=path,
            autotune_log=None,
            avg_cycles=cycles,
            samples=5,
            model=model,
            tags=tags,
            core=core,
            tokens=tokens,
        )


if __name__ == "__main__":
    unittest.main()
