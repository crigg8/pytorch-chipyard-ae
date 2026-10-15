#!/usr/bin/env python3
"""Check shared results and default resumption without Docker or FPGA access."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "docker/stage2"))
from local_runtime import collect_results


class SharedResultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stage2-results-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.scripts = self.project / "scripts"
        self.scripts.mkdir(parents=True)
        self.state = self.root / "state"
        self.results = self.scripts / "figures/results-workload"
        self.env = {"PATH": os.environ["PATH"], "HOME": str(self.root),
                    "STAGE2_STATE_ROOT": str(self.state)}

    def result(self, root, name, cycles=100, complete=True):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        model = directory / "model.log"
        model.write_text(f"Avg Model cycle: {cycles}\nModel samples: 1\n")
        os.utime(model, ns=(cycles, cycles))
        if complete:
            (directory / "autotune.log").write_text(f"autotune {cycles}\n")
        return directory

    def test_legacy_batches_merge_by_workload_and_keep_matching_newest_logs(self):
        first = self.state / "runs/first/results"
        second = self.state / "runs/second/results"
        third = self.state / "runs/failed/results"
        original = self.result(first, "squeezenet-scalar-4core")
        self.result(second, "squeezenet-scalar-4core", 200)
        self.result(third, "squeezenet-scalar-4core", 300, complete=False)
        self.result(third, "alexnet-gemmini-4core", 150)
        (third.parent / "run.json").write_text('{"attempts":[{"status":"failed"}]}')
        self.result(self.results, "alexnet-gemmini-4core", 400)
        self.assertEqual(collect_results(self.state, self.results), 1)
        self.assertEqual((self.results / "squeezenet-scalar-4core/autotune.log").read_text(), "autotune 200\n")
        self.assertIn("400", (self.results / "alexnet-gemmini-4core/model.log").read_text())
        self.assertIn("100", (original / "model.log").read_text())
        self.assertEqual(collect_results(self.state, self.results), 0)

    def test_plot_without_arguments_reads_the_combined_directory(self):
        for name in ("run-plot.sh", "env.sh"):
            shutil.copy2(REPO / "scripts" / name, self.scripts / name)
        support = self.project / "docker/stage2"
        support.mkdir(parents=True)
        shutil.copy2(REPO / "docker/stage2/local_runtime.py", support)
        figure = self.scripts / "figure"
        figure.mkdir()
        # Replace rendering only; exercise the real result parser on the path
        # selected by the public shell entrypoint.
        shutil.copy2(REPO / "scripts/figure/generate_plot_inputs.py", figure)
        (figure / "plot_results.sh").write_text('''#!/bin/bash
set -eu
python3 - "$2" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts/figure')
from generate_plot_inputs import discover_runs
print('SELECTED=' + str(Path(sys.argv[1])))
for run in discover_runs(Path(sys.argv[1])):
    print('WORKLOAD=' + run.workload)
PY
''')
        self.result(self.state / "runs/full/results", "squeezenet-scalar-4core")
        self.result(self.state / "runs/partial/results", "alexnet-gemmini-4core")
        self.env["PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR"] = "/old-checkout/results"
        result = subprocess.run(["bash", str(self.scripts / "run-plot.sh")], env=self.env,
                                text=True, capture_output=True, check=True)
        self.assertIn("SELECTED=" + str(self.results), result.stdout)
        for name in ("squeezenet-scalar-4core", "alexnet-gemmini-4core"):
            self.assertIn("WORKLOAD=" + name, result.stdout)
        explicit = self.root / "chosen results"
        self.result(explicit, "resnet50-gemmini-4core")
        result = subprocess.run(["bash", str(self.scripts / "run-plot.sh"), "--results-dir", str(explicit)],
                                env=self.env, text=True, capture_output=True, check=True)
        self.assertIn("WORKLOAD=resnet50-gemmini-4core", result.stdout)
        self.assertNotIn("WORKLOAD=alexnet", result.stdout)

    def test_new_runs_and_explicit_resume_share_the_same_result_mount(self):
        support = self.project / "docker/stage2"
        support.mkdir(parents=True)
        for name in ("local-launch.py", "local_runtime.py"):
            shutil.copy2(REPO / "docker/stage2" / name, support)
        (self.scripts / "env.sh").write_text(":\n")
        artifacts = self.project / "examples"
        artifacts.mkdir()
        db = self.root / "fpga.json"
        db.write_text('[{"bdf":"3f:00.0","uid":"fixture"}]')
        agent = socket.socket(socket.AF_UNIX)
        self.addCleanup(agent.close)
        agent_path = self.root / "agent.sock"
        agent.bind(str(agent_path))
        self.env.update(STAGE2_ARTIFACT_DIR=str(artifacts), FIRESIM_DB=str(db),
                        SSH_AUTH_SOCK=str(agent_path))
        for run_id, args in (("first", []), ("second", ["--experiment=figure-10", "--resume"]),
                             (None, ["--resume"]), ("rerun", ["--rerun-completed"])):
            env = dict(self.env)
            if run_id:
                env["RUN_ID"] = run_id
            proc = subprocess.run([sys.executable, str(support / "local-launch.py"),
                                   "--print-command", *args], env=env, text=True,
                                  capture_output=True, check=True)
            actual_id = next(line.split("=", 1)[1] for line in proc.stdout.splitlines()
                             if line.startswith("STAGE2_RUN_ID="))
            command = json.loads((self.state / "runs" / actual_id / "launch-command.json").read_text())
            self.assertIn(f"type=bind,src={self.results},dst={self.results}", command)
            self.assertIn("STAGE2_RESULTS_DIR=" + str(self.results), command)
            self.assertNotIn("--resume", command)
            self.assertEqual("--rerun-completed" in command, "--rerun-completed" in args)
        # An existing RUN_ID also resumes without an explicit flag.
        (self.state / "runs/first/run.json").write_text("{}")
        subprocess.run([sys.executable, str(support / "local-launch.py"), "--print-command"],
                       env=dict(self.env, RUN_ID="first"), capture_output=True, check=True)

    def workflow(self):
        shutil.copy2(REPO / "docker/stage2/workflow/run-stage2.sh", self.scripts)
        (self.scripts / "env.sh").write_text(":\n")
        self.env.update(PYTORCH_CHIPYARD_STAGE2_CONTAINER="1",
                        PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR=str(self.results),
                        PYTORCH_CHIPYARD_WORKLOAD_DIR=str(self.root / "workloads"),
                        FIRESIM_WORKLOAD_DIR=str(self.root / "installed"),
                        FIREMARSHAL_IMAGE_DIR=str(self.root / "images"),
                        TRACE=str(self.root / "trace"))
        (self.scripts / "package-firemarshal-workload.sh").write_text('''#!/bin/bash
set -eu
for workload in squeezenet-scalar-4core alexnet-gemmini-4core; do
  mkdir -p "$PYTORCH_CHIPYARD_WORKLOAD_DIR/overlay-$workload" "$FIRESIM_WORKLOAD_DIR"
  echo '{}' >"$PYTORCH_CHIPYARD_WORKLOAD_DIR/$workload.json"
  echo '{}' >"$FIRESIM_WORKLOAD_DIR/$workload.json"
done
''')
        (self.scripts / "build-firemarshal-images.sh").write_text('''#!/bin/bash
set -eu
workload=${1#--workload=}
echo "build $workload" >>"$TRACE"
mkdir -p "$FIREMARSHAL_IMAGE_DIR/$workload"
echo image >"$FIREMARSHAL_IMAGE_DIR/$workload/$workload.img"
echo boot >"$FIREMARSHAL_IMAGE_DIR/$workload/$workload-bin"
''')
        (self.scripts / "run-firesim-workloads.sh").write_text('''#!/bin/bash
set -eu
[[ $1 != --preflight-only ]] || exit 0
workload=${1#--workload=}
echo "run $workload" >>"$TRACE"
mkdir -p "$PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR/$workload"
echo model >"$PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR/$workload/model.log"
echo autotune >"$PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR/$workload/autotune.log"
''')

    def run_workflow(self, *args):
        proc = subprocess.run(["bash", str(self.scripts / "run-stage2.sh"), *args],
                              env=self.env, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return proc.stdout

    def test_default_skips_log_pairs_runs_incomplete_and_honors_forced_rerun(self):
        self.workflow()
        self.result(self.results, "squeezenet-scalar-4core")
        self.result(self.results, "alexnet-gemmini-4core", complete=False)
        self.run_workflow()
        trace = self.root / "trace"
        first = trace.read_text()
        self.assertEqual(first, "build alexnet-gemmini-4core\nrun alexnet-gemmini-4core\n")
        self.assertIn("no pending FireSim workloads", self.run_workflow())
        self.assertEqual(trace.read_text(), first)
        self.run_workflow("--rerun-completed")
        self.assertEqual(trace.read_text().count("run squeezenet-scalar-4core"), 1)
        self.assertEqual(trace.read_text().count("run alexnet-gemmini-4core"), 2)

    def test_experiment_accepts_resume_and_needs_no_marker_or_image(self):
        self.workflow()
        for model in ("alexnet", "mobilenetv2", "resnet50", "squeezenet"):
            for backend in ("gemmini", "gemmini-im2col"):
                self.result(self.results, f"{model}-{backend}-4core")
        output = self.run_workflow("--experiment=figure-10", "--resume")
        self.assertIn("8 complete, 0 pending", output)
        self.assertFalse((self.root / "trace").exists())


if __name__ == "__main__":
    unittest.main()
