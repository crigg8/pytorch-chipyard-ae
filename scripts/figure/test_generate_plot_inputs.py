import importlib.util
import json
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


class Im2colAttributionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.previous_artifact_root = MODULE.ARTIFACT_ROOT
        MODULE.ARTIFACT_ROOT = self.root / "examples"

    def tearDown(self) -> None:
        MODULE.ARTIFACT_ROOT = self.previous_artifact_root
        self.temp_dir.cleanup()

    def test_uses_launch_weight_shapes_and_accounts_for_model_overhead(self) -> None:
        artifact = self.root / "examples" / "artifact-resnet50" / "gemmini"
        artifact.mkdir(parents=True)
        buffers = [
            {"name": "weight7", "size_hint": [64, 3, 7, 7]},
            {"name": "weight1", "size_hint": [64, 64, 1, 1]},
        ]
        steps = [
            {
                "kind": "launch",
                "kernel_name": "triton_poi_fused_convolution_0",
                "triton_meta": {},
                "call_args": [{"kind": "constant", "name": "weight7"}],
            },
            {
                "kind": "launch",
                "kernel_name": "triton_tem_fused_convolution_1",
                "triton_meta": {
                    "chipyard_default_kernel_name": "triton_convolution2d_default"
                },
                "call_args": [{"kind": "graph_input", "name": "input"}],
            },
            {
                "kind": "launch",
                "kernel_name": "triton_tem_fused_convolution_2",
                "triton_meta": {
                    "chipyard_default_kernel_name": "triton_mm_chipyard_default"
                },
                "call_args": [{"kind": "constant", "name": "weight1"}],
            },
            {
                "kind": "launch",
                "kernel_name": "triton_poi_fused_relu_3",
                "triton_meta": {},
                "call_args": [],
            },
        ]
        (artifact / "model_spec.json").write_text(
            json.dumps({"buffers": buffers, "steps": steps})
        )

        result_dir = self.root / "results" / "resnet50-gemmini-4core"
        result_dir.mkdir(parents=True)
        model_path = result_dir / "model.log"
        model_path.write_text(
            "Model Launch Log\n\n"
            "Avg Model cycle: 70000000\n"
            "Model samples: 1\n\n"
            "All Kernel Cycle Stats (execution order)\n"
            "1. prepack7\nTotal launch cycle: 10000000\n"
            "2. selected_conv7\nTotal launch cycle: 30000000\n"
            "3. selected_mm1\nTotal launch cycle: 20000000\n"
            "4. pointwise\nTotal launch cycle: 5000000\n"
        )
        autotune_path = result_dir / "autotune.log"
        autotune_path.write_text(
            "Autotune log\n"
            "Autotune Candidate\n"
            "Kernel: selected_conv7\n"
            "Raw kernel: triton_convolution2d_selected\n"
            "  KERNEL_H=7\n"
            "  KERNEL_W=7\n"
            "  STRIDE_H=2\n"
            "  PADDING_H=3\n"
        )
        run = MODULE.WorkloadRun(
            workload="resnet50-gemmini-4core",
            result_dir=result_dir,
            model_log=model_path,
            autotune_log=autotune_path,
            avg_cycles=70000000,
            samples=1,
            model="resnet50",
            tags=("gemmini",),
            core=4,
            tokens=None,
        )

        totals = MODULE.cycles_by_im2col_label(
            run, ["7x7 conv", "1x1 conv", "3x3 conv", "prepack", "others"]
        )
        self.assertEqual(
            totals,
            {
                "7x7 conv": 30.0,
                "1x1 conv": 20.0,
                "3x3 conv": 0.0,
                "prepack": 10.0,
                "others": 10.0,
            },
        )

    def test_classifies_im2col_weight_shapes(self) -> None:
        resnet_launch = MODULE.Im2colLaunch(
            raw="triton_mm_chipyard",
            constant_shapes=((64, 64, 3, 3),),
            has_graph_input=False,
        )
        squeeze_first = MODULE.Im2colLaunch(
            raw="triton_mm_chipyard",
            constant_shapes=((64, 3, 3, 3),),
            has_graph_input=True,
        )
        squeeze_fire = MODULE.Im2colLaunch(
            raw="triton_mm_chipyard",
            constant_shapes=((64, 16, 3, 3),),
            has_graph_input=False,
        )

        self.assertEqual(
            MODULE.im2col_label("ResNet50", "selected_mm", None, resnet_launch),
            "3x3 conv",
        )
        self.assertEqual(
            MODULE.im2col_label("SqueezeNet", "selected_mm", None, squeeze_first),
            "3x3 s2p0",
        )
        self.assertEqual(
            MODULE.im2col_label("SqueezeNet", "selected_mm", None, squeeze_fire),
            "Fire 3x3",
        )

    def test_classifies_resnet_constant_convolution_helpers_as_prepack(self) -> None:
        batch_norm_helper = MODULE.Im2colLaunch(
            raw="",
            constant_shapes=((64,),),
            has_graph_input=False,
        )

        self.assertEqual(
            MODULE.im2col_label(
                "ResNet50",
                "triton_poi_fused_convolution_native_batch_norm_1",
                None,
                batch_norm_helper,
            ),
            "prepack",
        )
        self.assertEqual(
            MODULE.im2col_label(
                "SqueezeNet",
                "triton_poi_fused_convolution_relu_1",
                None,
                batch_norm_helper,
            ),
            "others",
        )


if __name__ == "__main__":
    unittest.main()
