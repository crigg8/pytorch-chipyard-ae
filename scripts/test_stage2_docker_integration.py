#!/usr/bin/env python3
"""Check the v1.0.0 integration without Docker or FPGA access."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / "docker/stage2/workflow"
WORKLOADS = ("smoke-gemm-rvv-4core", "smoke-gemm-gemmini-4core", "smoke-gemm-scalar-16core")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage2IntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stage2-integration-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def inputs(self):
        artifacts = self.root / "inputs"
        for backend, cores in (("rvv", 4), ("gemmini", 4), ("scalar", 16)):
            directory = artifacts / backend
            directory.mkdir(parents=True)
            for name in (f"model-{cores}core.elf", "input.bin", "weights.bin", "model_spec.json"):
                (directory / name).write_text("fixture")
            (directory / ".pytorch-chipyard-workload-rel").write_text("smoke-gemm/" + backend)
        return artifacts

    def test_image_installs_validated_workflow_without_replacing_stage1_scripts(self):
        module = load("image_finalizer", REPO / "docker/stage2/finalize-image.py")
        project = self.root / "project"
        shutil.copytree(REPO / "scripts", project / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
        module.install_container_workflow(WORKFLOW, project)
        for name in ("env.sh", "run-stage2.sh", "run-firesim-workloads.sh",
                     "build-firemarshal-images.sh", "package-firemarshal-workload.sh"):
            self.assertEqual((project / "scripts" / name).read_bytes(), (WORKFLOW / name).read_bytes())
        for name in ("install.sh", "build-chipyard-elves.sh", "run-stage1.sh", "run-smoke-test-stage1.sh"):
            self.assertEqual((project / "scripts" / name).read_bytes(), (REPO / "scripts" / name).read_bytes())

    def test_tampered_workflow_cannot_enter_build_context(self):
        module = load("image_finalizer", REPO / "docker/stage2/finalize-image.py")
        workflow = self.root / "docker/stage2/workflow"
        shutil.copytree(WORKFLOW, workflow)
        with (workflow / "env.sh").open("a") as stream:
            stream.write("\n# modified after validation\n")
        project = self.root / "project"
        (project / "scripts").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            module.install_container_workflow(workflow, project)

    def docker_recorder(self):
        directory = self.root / "bin"
        directory.mkdir()
        docker = directory / "docker"
        docker.write_text("#!" + sys.executable + "\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == 'info':
    print('["name=rootless"]')
elif args[:2] == ['image', 'inspect']:
    print('sha256:' + 'a' * 64)
elif args[0] == 'run':
    source = sys.stdin.read()
    Path(os.environ['DOCKER_RECORD']).write_text(json.dumps({'args': args, 'stdin': source}))
else:
    raise SystemExit('unexpected Docker call: ' + repr(args))
''')
        docker.chmod(0o755)
        return dict(os.environ, PATH=str(directory) + os.pathsep + os.environ["PATH"],
                    DOCKER_RECORD=str(self.root / "docker.json"))

    def test_input_preparation_mounts_only_artifacts_and_uses_the_bundled_compiler(self):
        env = self.docker_recorder()
        artifacts = self.root / "inputs"
        artifacts.mkdir()
        result = subprocess.run(["bash", str(REPO / "scripts/prepare-stage2-inputs.sh")],
                                env=dict(env, STAGE2_ARTIFACT_DIR=str(artifacts)),
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads((self.root / "docker.json").read_text())
        mounts = [record["args"][i + 1] for i, arg in enumerate(record["args"][:-1]) if arg == "--mount"]
        self.assertEqual(len(mounts), 1, mounts)
        self.assertIn("src=" + str(artifacts), mounts[0])
        self.assertIn("/opt/riscv-gcc-12.2.0/riscv-tools/bin/riscv64-unknown-linux-gnu-g++", record["stdin"])
        self.assertIn("--chipyard-env=/opt/riscv-gcc-12.2.0/env.sh", record["stdin"])

    def test_stage1_links_workloads_before_reporting_success(self):
        project = self.root / "project"
        scripts = project / "scripts"
        scripts.mkdir(parents=True)
        toolchain = self.root / "toolchain"
        compiler = toolchain / "riscv-tools/bin/riscv64-unknown-linux-gnu-g++"
        compiler.parent.mkdir(parents=True)
        compiler.write_text('#!/bin/bash\nprintf "%s\\n" "${GCC_VERSION:-12.2.0}"\n')
        compiler.chmod(0o755)
        source = (REPO / "scripts/run-stage1.sh").read_text()
        (scripts / "run-stage1.sh").write_text(source.replace("/opt/riscv-gcc-12.2.0", str(toolchain)))
        # Execute the real orchestration, replacing expensive compilation and
        # linking with a recorder that can also fail at the linker boundary.
        for name in ("run-cnn.sh", "run-alias-first-cnn.sh", "run-alias-first-sdpa.sh",
                     "run-im2col.sh", "run-sdpa.sh", "run-flex-attn.sh",
                     "run_gemmini_autotune.sh", "run_table4.sh", "build-chipyard-elves.sh"):
            (scripts / name).write_text('''
set -euo pipefail
printf '%s %s\\n' "${0##*/}" "$*" >> "$STAGE1_RECORD"
for arg in "$@"; do
  case "$arg" in --output-dir=*) mkdir -p "${arg#*=}";; esac
done
if [[ "${0##*/}" == build-chipyard-elves.sh ]]; then
  exit "${LINK_STATUS:-0}"
fi
''')
        skip_all = ["--skip-cnn", "--skip-alias-first", "--skip-im2col", "--skip-sdpa",
                    "--skip-flex-attn", "--skip-gemmini-autotune", "--skip-table4"]
        cases = (
            ("full", [], {}, True, 0),
            ("simple", [], {"PYTORCH_CHIPYARD_SIMPLE_STAGE2": "1"}, True, 0),
            ("alias", ["--only-alias-first"], {}, True, 0),
            ("cnn-alias", ["--only-alias-first-cnn-off"], {}, True, 0),
            ("table4", ["--only-table4"], {}, False, 0),
            ("skip-all", skip_all, {}, False, 0),
            ("link-failure", ["--skip-table4"], {"LINK_STATUS": "23"}, True, 23),
            ("wrong-compiler", ["--skip-table4"], {"GCC_VERSION": "11.4.0"}, False, 1),
        )
        for name, args, overrides, links, status in cases:
            with self.subTest(case=name):
                record = self.root / (name + ".txt")
                env = dict(os.environ, STAGE1_RECORD=str(record), LINK_STATUS="0",
                           GCC_VERSION="12.2.0", PYTORCH_CHIPYARD_SIMPLE_STAGE2="0",
                           TABLE4_RESULTS_ROOT=str(project / "results/table4"))
                env.update(overrides)
                result = subprocess.run(["bash", str(scripts / "run-stage1.sh"), *args],
                                        env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, status, result.stdout + result.stderr)
                self.assertEqual("STAGE1_STATUS=PASS" in result.stdout, status == 0)
                calls = record.read_text().splitlines() if record.exists() else []
                link_calls = [line for line in calls if line.startswith("build-chipyard-elves.sh ")]
                self.assertEqual(len(link_calls), int(links), calls)
                if links:
                    self.assertEqual(calls[-1], link_calls[0])
                    self.assertIn("--chipyard-env=" + str(toolchain / "env.sh"), link_calls[0])
                    self.assertIn("--riscv-gxx=" + str(compiler), link_calls[0])

    def test_stage1_smoke_requires_all_linked_elfs_before_reporting_success(self):
        project = self.root / "project"
        scripts = project / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(REPO / "scripts/run-smoke-test-stage1.sh", scripts)
        # Replace the expensive compiler boundary, while executing the real
        # Stage 1 orchestration and its completion/failure checks.
        (scripts / "stage1.sh").write_text('''
pc_die() { echo "$*" >&2; exit 1; }
pc_usage_error() { pc_die "$@"; }
pc_prepare_environment() { :; }
pc_write_artifact_workload_hint() {
  mkdir -p "$1"
  printf '%s\\n' "$2" > "$1/.pytorch-chipyard-workload-rel"
}
pc_write_artifact_build_plan() { :; }
pc_run_compile_once() {
  for name in input.bin weights.bin model_spec.json; do
    printf 'fixture\\n' > "$2/$name"
  done
}
pc_require_artifacts() { test -s "$1/input.bin"; }
''')
        (scripts / "build-chipyard-elves.sh").write_text('''
set -euo pipefail
printf '%s\\n' "$@" > "$LINK_RECORD"
[[ "$LINK_MODE" != fail ]] || exit 23
for arg in "$@"; do
  case "$arg" in
    --artifact-dir=*)
      directory="${arg#*=}"
      case "${directory##*/}" in scalar) cores=16;; *) cores=4;; esac
      [[ "$LINK_MODE" != missing || "${directory##*/}" != rvv ]] || continue
      printf 'fixture ELF\\n' > "$directory/model-${cores}core.elf"
      ;;
  esac
done
''')
        for mode in ("success", "fail", "missing"):
            with self.subTest(link_mode=mode):
                artifacts = project / "results" / mode
                record = self.root / (mode + "-link.txt")
                result = subprocess.run(
                    ["bash", str(scripts / "run-smoke-test-stage1.sh"), "--artifact-root=" + str(artifacts)],
                    env=dict(os.environ, LINK_MODE=mode, LINK_RECORD=str(record)),
                    text=True, capture_output=True)
                if mode == "success":
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("SMOKE_STAGE1_STATUS=PASS", result.stdout)
                    self.assertIn("--chipyard-env=/opt/riscv-gcc-12.2.0/env.sh", record.read_text())
                    self.assertIn("--riscv-gxx=/opt/riscv-gcc-12.2.0/riscv-tools/bin/riscv64-unknown-linux-gnu-g++", record.read_text())
                    for backend, cores in (("rvv", 4), ("gemmini", 4), ("scalar", 16)):
                        self.assertGreater((artifacts / backend / f"model-{cores}core.elf").stat().st_size, 0)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("SMOKE_STAGE1_STATUS=PASS", result.stdout)
                    if mode == "missing":
                        self.assertIn("missing linked smoke ELF", result.stderr)

    def test_public_smoke_entrypoint_selects_only_three_container_workloads(self):
        artifacts = self.inputs()
        agent = socket.socket(socket.AF_UNIX)
        self.addCleanup(agent.close)
        agent.bind(str(self.root / "agent"))
        database = self.root / "fpga.json"
        database.write_text(json.dumps([{"bdf": "3f:00.0", "uid": "fixture"}]))
        env = dict(os.environ, STAGE2_SMOKE_ARTIFACT_DIR=str(artifacts),
                   STAGE2_ARTIFACT_DIR="/nonexistent/full-models", FIRESIM_DB=str(database),
                   STAGE2_STATE_ROOT=str(self.root / "state"), RUN_ID="fixture",
                   SSH_AUTH_SOCK=str(self.root / "agent"))
        result = subprocess.run(["bash", str(REPO / "scripts/run-smoke-test-stage2-docker.sh"),
                                 "--print-command"], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        command = json.loads((self.root / "state/runs/fixture/launch-command.json").read_text())
        self.assertIn("--workload=" + ",".join(WORKLOADS), command)
        self.assertIn(f"type=bind,src={artifacts},dst=/opt/pytorch-chipyard/examples,readonly", command)
        self.assertIn("PYTORCH_CHIPYARD_OMP_FIRST_CPU_SMOKE_GEMM_RVV_4CORE=0", command)
        self.assertNotIn("SMOKE_TEST_STATUS=PASS", result.stdout)

    def test_packaged_smoke_retains_validated_rvv_affinity_and_rejects_corruption(self):
        artifacts = self.inputs()
        project = self.root / "project"
        shutil.copytree(WORKFLOW, project / "scripts")
        (project / "examples").mkdir()
        env = dict(os.environ, PYTORCH_CHIPYARD_WORKLOAD_DIR=str(self.root / "workloads"),
                   FIRESIM_WORKLOAD_DIR=str(self.root / "deploy"),
                   FIREMARSHAL_IMAGE_DIR=str(self.root / "images"),
                   PYTORCH_CHIPYARD_OMP_FIRST_CPU_SMOKE_GEMM_RVV_4CORE="0")
        result = subprocess.run(["bash", str(project / "scripts/package-firemarshal-workload.sh"),
                                 "--no-chipyard-check",
                                 *("--artifact-dir=" + str(artifacts / b) for b in ("rvv", "gemmini", "scalar"))],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for workload in WORKLOADS:
            self.assertTrue((self.root / "workloads" / (workload + ".json")).is_file())
            image = self.root / "images" / workload
            image.mkdir(parents=True)
            (image / (workload + ".img")).touch()
            (image / (workload + "-bin")).touch()
        source = (WORKFLOW / "run-firesim-workloads.sh").read_text()
        functions = "set -euo pipefail\n" + "\n".join(
            re.search(r"^" + name + r"\(\) \{\n.*?^\}\n", source, re.MULTILINE | re.DOTALL).group()
            for name in ("die", "require_file", "contains_tag", "rvv_success_profile_for", "validate_rvv_success_package"))
        command = functions + '\nfor workload in "$@"; do validate_rvv_success_package "$workload"; done\n'
        def validate():
            return subprocess.run(["bash", "-c", command, "test", *WORKLOADS],
                                  env=env, text=True, capture_output=True)
        self.assertEqual(validate().returncode, 0)
        runner = next((self.root / "workloads/overlay-smoke-gemm-rvv-4core").rglob("run_*.sh"))
        original = runner.read_text()
        self.assertIn('OMP_PLACES="{0},{1},{2},{3}"', original)
        runner.write_text(original.replace('OMP_PLACES="{0},{1},{2},{3}"', 'OMP_PLACES="{4},{5},{6},{7}"'))
        result = validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not use validated OMP_PLACES", result.stderr)

    def test_delivery_records_archive_and_installer_launcher_hashes(self):
        module = load("image_export", REPO / "scripts/export-stage2-local-image.py")
        archive = self.root / "image.tar"
        archive.write_bytes(b"archive fixture")
        record = module.delivery_record({"Id": "sha256:" + "a" * 64}, archive)
        self.assertEqual(record["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(record["archive"], "image.tar")
        for name, digest in record["launcher_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((REPO / "docker/stage2" / name).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
