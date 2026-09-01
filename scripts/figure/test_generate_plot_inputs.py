import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("generate_plot_inputs.py")
SPEC = importlib.util.spec_from_file_location("generate_plot_inputs_tested", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def model_log(cycles: int) -> str:
    return (
        "Model Launch Log\n\n"
        f"Avg Model cycle: {cycles}\n"
        "Model samples: 1\n"
    )


class CompatibilityLogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.results = self.root / "results"
        self.logs = self.root / "examples" / ".logs"
        self.csv_dir = self.root / "csv"
        self.logs.mkdir(parents=True)
        MODULE.LOG_DIR = self.logs
        MODULE.CSV_DIR = self.csv_dir

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_result(self, workload: str, cycles: int, mtime_ns: int) -> Path:
        result_dir = self.results / workload
        result_dir.mkdir(parents=True)
        model_path = result_dir / "model.log"
        autotune_path = result_dir / "autotune.log"
        model_path.write_text(model_log(cycles))
        autotune_path.write_text("autotune\n")
        os.utime(model_path, ns=(mtime_ns, mtime_ns))
        os.utime(autotune_path, ns=(mtime_ns, mtime_ns))
        return model_path

    def test_preserves_session_logs_and_keeps_newest_copy(self) -> None:
        source = self.add_result("mobilenetv2-scalar-4core", 100, 100)
        session_log = self.logs / "stage2-resume-20260829T102330Z.log"
        session_log.write_text("keep me\n")
        destination = self.logs / "mobilenet-scalar-4core-model.log"
        destination.write_text(model_log(200))
        os.utime(destination, ns=(200, 200))

        MODULE.reset_generated_csvs()
        MODULE.prepare_compat_logs(MODULE.discover_runs(self.results))
        self.assertEqual(session_log.read_text(), "keep me\n")
        self.assertIn("200", destination.read_text())

        source.write_text(model_log(300))
        os.utime(source, ns=(300, 300))
        MODULE.prepare_compat_logs(MODULE.discover_runs(self.results))
        self.assertIn("300", destination.read_text())
        self.assertTrue(session_log.exists())

    def test_loads_latest_alias_once_from_examples(self) -> None:
        older = self.logs / "mobilenet-scalar-4core-model.log"
        newer = self.logs / "mobilenetv2-scalar-4core-model.log"
        older.write_text(model_log(100))
        newer.write_text(model_log(200))
        os.utime(older, ns=(100, 100))
        os.utime(newer, ns=(200, 200))

        runs = MODULE.discover_compat_runs(self.logs)
        selected = [
            run
            for run in runs
            if run.workload == "mobilenetv2-scalar-4core"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].avg_cycles, 200)

    def test_normalizes_legacy_llm_scalar_alias(self) -> None:
        rocket = self.logs / "opt-rocket-gemmini-flash-256tok-4core-model.log"
        scalar = self.logs / "opt-scalar-gemmini-flash-256tok-4core-model.log"
        rocket.write_text(model_log(100))
        scalar.write_text(model_log(200))
        os.utime(rocket, ns=(100, 100))
        os.utime(scalar, ns=(200, 200))

        runs = MODULE.discover_compat_runs(self.logs)
        selected = [
            run
            for run in runs
            if run.workload == "opt-rocket-gemmini-flash-256tok-4core"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].avg_cycles, 200)


if __name__ == "__main__":
    unittest.main()
