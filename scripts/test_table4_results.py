#!/usr/bin/env python3

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import table4_results


KERNEL_ID = "resnet50_classifier"


class TestTable4Results(unittest.TestCase):
    def test_builtin_kernels_use_mobilenetv2_classifier(self):
        kernels = table4_results.table4_kernels()
        self.assertEqual(
            [kernel["id"] for kernel in kernels],
            [
                "squeezenet_fire2_squeeze",
                "resnet50_classifier",
                "mobilenetv2_classifier",
            ],
        )
        mobilenet = table4_results.get_kernel("mobilenetv2_classifier")
        self.assertEqual(
            (mobilenet["m"], mobilenet["n"], mobilenet["k"], mobilenet["macs"]),
            (1, 1000, 1280, 1280000),
        )

    def append(
        self,
        raw_path: Path,
        *,
        trial: str,
        toolchain: str,
        phase: str,
        simulator: str = "",
        wall: str = "1",
        status: str = "PASS",
        exit_code: str = "0",
        kernel_id: str = KERNEL_ID,
    ) -> None:
        kernel = table4_results.get_kernel(kernel_id)
        table4_results.append_row(
            SimpleNamespace(
                csv=str(raw_path),
                run_id="test",
                trial=trial,
                kernel_id=kernel_id,
                kernel_label=kernel["label"],
                source_model=kernel["source_model"],
                shape=f"{kernel['m']}x{kernel['n']}x{kernel['k']}",
                macs=str(kernel["macs"]),
                toolchain=toolchain,
                phase=phase,
                simulator=simulator,
                total_wall_s=wall,
                status=status,
                exit_code=exit_code,
                artifact_path="artifact",
                log_path="log",
                target_metadata="target",
                notes="",
            )
        )

    def summary_row(self, raw_path: Path, output_path: Path) -> dict[str, str]:
        table4_results.summarize(raw_path, output_path)
        with output_path.open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        return next(row for row in rows if row["kernel_id"] == KERNEL_ID)

    def test_extract_firesim_wall_requires_pass(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uart = root / "uartlog"
            uart.write_text(
                "Simulation complete.\n"
                "*** PASSED *** after 123 cycles\n"
                "Wallclock Time Elapsed: 981.2 s\n"
            )
            self.assertEqual(table4_results.extract_firesim_wall(uart), "981.2")

            failed = root / "failed-uartlog"
            failed.write_text("Wallclock Time Elapsed: 1.0 s\n")
            with self.assertRaises(SystemExit):
                table4_results.extract_firesim_wall(failed)

    def test_summary_uses_successful_medians_for_four_measurements(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.csv"
            table4_results.init_csv(raw_path)
            measurements = [
                ("PyTorch-Chipyard", "compile", "", "10", "14"),
                ("PyTorch-Chipyard", "rtl", "firesim", "20", "24"),
                ("TVM-Gemmini", "compile", "", "2", "4"),
                ("TVM-Gemmini", "rtl", "verilator", "100", "140"),
            ]
            for toolchain, phase, simulator, first, second in measurements:
                self.append(
                    raw_path,
                    trial="1",
                    toolchain=toolchain,
                    phase=phase,
                    simulator=simulator,
                    wall=first,
                )
                self.append(
                    raw_path,
                    trial="2",
                    toolchain=toolchain,
                    phase=phase,
                    simulator=simulator,
                    wall=second,
                )

            row = self.summary_row(raw_path, root / "table4.csv")
            self.assertEqual(row["pytorch_compile_s"], "12.000000")
            self.assertEqual(row["firesim_s"], "22.000000")
            self.assertEqual(row["tvm_compile_s"], "3.000000")
            self.assertEqual(row["verilator_s"], "120.000000")
            self.assertEqual(row["status"], "PASS")

    def test_successful_retry_supersedes_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.csv"
            table4_results.init_csv(raw_path)
            measurements = [
                ("PyTorch-Chipyard", "compile", "", "10"),
                ("PyTorch-Chipyard", "rtl", "firesim", "20"),
                ("TVM-Gemmini", "compile", "", "2"),
                ("TVM-Gemmini", "rtl", "verilator", "100"),
            ]
            for toolchain, phase, simulator, wall in measurements:
                self.append(
                    raw_path,
                    trial="1",
                    toolchain=toolchain,
                    phase=phase,
                    simulator=simulator,
                    wall=wall,
                )
            self.append(
                raw_path,
                trial="1",
                toolchain="TVM-Gemmini",
                phase="rtl",
                simulator="verilator",
                wall="",
                status="FAIL",
                exit_code="19",
            )
            failed = self.summary_row(raw_path, root / "failed.csv")
            self.assertEqual(failed["status"], "FAIL")

            self.append(
                raw_path,
                trial="1",
                toolchain="TVM-Gemmini",
                phase="rtl",
                simulator="verilator",
                wall="120",
            )
            recovered = self.summary_row(raw_path, root / "recovered.csv")
            self.assertEqual(recovered["status"], "PASS")
            self.assertEqual(recovered["verilator_s"], "120.000000")

    def test_has_passed_measurement_matches_exact_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "raw.csv"
            table4_results.init_csv(raw_path)
            self.append(
                raw_path,
                trial="1",
                toolchain="PyTorch-Chipyard",
                phase="compile",
            )
            self.assertTrue(
                table4_results.has_passed_measurement(
                    raw_path,
                    trial="1",
                    kernel_id=KERNEL_ID,
                    toolchain="PyTorch-Chipyard",
                    phase="compile",
                    simulator="",
                )
            )
            self.assertFalse(
                table4_results.has_passed_measurement(
                    raw_path,
                    trial="1",
                    kernel_id=KERNEL_ID,
                    toolchain="PyTorch-Chipyard",
                    phase="rtl",
                    simulator="firesim",
                )
            )

    def test_tvm_compile_metadata_rejects_old_timing_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "table4-metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "compile_scope": "relay.build",
                    }
                )
            )
            self.assertFalse(
                table4_results.has_current_tvm_compile_metadata(metadata_path)
            )

            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "compile_scope": table4_results.TVM_COMPILE_SCOPE,
                        "compile_excludes": table4_results.TVM_COMPILE_EXCLUDES,
                    }
                )
            )
            self.assertTrue(
                table4_results.has_current_tvm_compile_metadata(metadata_path)
            )

    def test_latex_uses_seconds_and_na(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.csv"
            summary_path = root / "table4.csv"
            output_path = root / "rows.tex"
            table4_results.init_csv(raw_path)
            self.append(
                raw_path,
                trial="1",
                toolchain="PyTorch-Chipyard",
                phase="compile",
                wall="1.25",
            )
            table4_results.summarize(raw_path, summary_path)
            table4_results.write_latex(summary_path, output_path)
            text = output_path.read_text()
            self.assertIn(r"1.2\,s", text)
            self.assertIn("N/A", text)

    def test_validate_summary_requires_complete_positive_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.csv"
            summary_path = root / "table4.csv"
            table4_results.init_csv(raw_path)
            measurements = [
                ("PyTorch-Chipyard", "compile", "", "10"),
                ("PyTorch-Chipyard", "rtl", "firesim", "20"),
                ("TVM-Gemmini", "compile", "", "2"),
                ("TVM-Gemmini", "rtl", "verilator", "100"),
            ]
            for kernel in table4_results.table4_kernels():
                for toolchain, phase, simulator, wall in measurements:
                    self.append(
                        raw_path,
                        trial="1",
                        kernel_id=str(kernel["id"]),
                        toolchain=toolchain,
                        phase=phase,
                        simulator=simulator,
                        wall=wall,
                    )
            table4_results.summarize(raw_path, summary_path)
            table4_results.validate_summary(summary_path, minimum_trials=1)

            with summary_path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            rows[0]["firesim_s"] = ""
            with summary_path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=table4_results.SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(SystemExit):
                table4_results.validate_summary(summary_path, minimum_trials=1)


if __name__ == "__main__":
    unittest.main()
