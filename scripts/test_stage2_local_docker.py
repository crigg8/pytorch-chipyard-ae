#!/usr/bin/env python3
"""Regression checks for full-container execution without changing host hardware."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import hashlib
import io
import signal
import tempfile
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SUPPORT = REPO / "docker/stage2"
sys.path.insert(0, str(SUPPORT))
import local_runtime


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SUPPORT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("local_launch", "local-launch.py")
snapshot = load("patch_local", "patch-local.py")
privileged = load("privileged_smoke", "privileged-smoke.py")
receiver = load("host_receiver", "host-program.py")
import host_programmer


class LocalDockerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stage2-local-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        agent = socket.socket(socket.AF_UNIX)
        self.addCleanup(agent.close)
        self.agent = self.root / "agent.sock"
        agent.bind(str(self.agent))
        self.db = self.root / "fpga.json"
        self.db.write_text(json.dumps([{"bdf": "3f:00.0", "uid": "fixture-jtag"}]))

    def test_bitstream_receiver_rejects_truncation_corruption_and_oversize(self):
        payload = bytes(range(256)) * 1024
        digest = hashlib.sha256(payload).hexdigest()
        target = self.root / "valid.bit"
        receiver.receive(io.BytesIO(payload), target, len(payload), digest)
        self.assertEqual(target.read_bytes(), payload)
        for name, data, size, expected in (
                ("truncated", payload[:-1], len(payload), "truncated"),
                ("corrupt", payload[::-1], len(payload), "mismatch"),
                ("extra", payload+b"x", len(payload), "mismatch"),
                ("oversize", b"", 257*1024*1024, "size")):
            with self.assertRaisesRegex(RuntimeError, expected):
                receiver.receive(io.BytesIO(data), self.root / name, size, digest)

    def test_programming_wrong_bdf_never_contacts_host(self):
        with patch.object(host_programmer, "configuration", return_value={"bdf": "3f:00.0"}), \
                patch.object(host_programmer, "request") as request:
            with self.assertRaisesRegex(RuntimeError, "different FPGA"):
                host_programmer.program(self.root / "nonexistent.bit", "af:00.0")
            request.assert_not_called()

    def test_agent_rejects_regular_file_symlink_and_other_owner(self):
        self.assertEqual(local_runtime.agent_socket(self.agent, os.getuid()), self.agent)
        regular = self.root / "file"
        regular.write_text("not a socket")
        link = self.root / "link"
        link.symlink_to(self.agent)
        for path, uid in ((regular, os.getuid()), (link, os.getuid()), (self.agent, os.getuid()+1)):
            with self.assertRaisesRegex(RuntimeError, "socket owned"):
                local_runtime.agent_socket(path, uid)

    def hardware(self):
        import yaml
        archive = self.root / "bitstream.tar.gz"
        archive.write_bytes(b"bitstream fixture")
        driver = self.root / "driver.tar.gz"
        driver.write_bytes(b"driver fixture")
        source = self.root / "hwdb.yaml"
        source.write_text(yaml.safe_dump({"fixture": {"bitstream_tar": archive.as_uri(),
                                                     "driver_tar": "driver.tar.gz"}}))
        return source, archive, driver

    def test_hwdb_rewrites_archives_and_hashes_all_runtime_inputs(self):
        import yaml
        source, archive, driver = self.hardware()
        before = source.read_bytes()
        output = self.root / "container.yaml"
        mounts = launcher.prepare_hwdb(source, output)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual({s for s, _ in mounts}, {str(archive), str(driver)})
        document = yaml.safe_load(output.read_text())
        self.assertTrue(document["fixture"]["driver_tar"].startswith("file:///opt/stage2-hardware/"))
        # Hash the same payload through filesystem paths available in this test.
        document["fixture"]["bitstream_tar"] = archive.as_uri()
        document["fixture"]["driver_tar"] = driver.as_uri()
        output.write_text(yaml.safe_dump(document))
        with patch.dict(os.environ, FIRESIM_HWDB_PATH=str(output), FIRESIM_DB=str(self.db)):
            first = local_runtime.hardware_identity()
            driver.write_bytes(b"changed driver")
            self.assertNotEqual(first, local_runtime.hardware_identity())
            first = local_runtime.hardware_identity()
            self.db.write_text(json.dumps([{"bdf": "af:00.0", "uid": "different-card"}]))
            self.assertNotEqual(first, local_runtime.hardware_identity())

    def test_remote_hwdb_does_not_silently_skip_content_pinning(self):
        source = self.root / "hwdb.yaml"
        source.write_text("fixture:\n  bitstream_tar: https://example.test/mutable.tar.gz\n")
        with self.assertRaisesRegex(ValueError, "download HWDB archives locally"):
            launcher.prepare_hwdb(source, self.root / "output.yaml")

    def fake_sysfs(self):
        root = self.root / "sys"
        pci = root / "bus/pci/devices/0000:3f:00.0"
        pci.mkdir(parents=True)
        (pci / "vendor").write_text("0x10ee\n")
        driver = root / "bus/pci/drivers/xdma"
        driver.mkdir(parents=True)
        (pci / "driver").symlink_to(driver)
        (root / "module/xdma").mkdir(parents=True)
        for name, number in (("user", "235:0"), ("h2c_0", "235:32"), ("c2h_0", "235:36")):
            item = pci / "xdma" / ("xdma3_" + name)
            item.mkdir(parents=True)
            (item / "dev").write_text(number)
        return root

    def test_device_discovery_follows_selected_bdf_and_refreshes_stale_nodes(self):
        root = self.fake_sysfs()
        dev = self.root / "dev"
        dev.mkdir()
        (dev / "xdma3_user").write_text("stale")
        with patch.object(local_runtime.os, "mknod") as create:
            _, bdf, numbers = local_runtime.sync_devices(self.db, root, dev)
        self.assertEqual(bdf, "0000:3f:00.0")
        self.assertEqual(numbers["xdma3_h2c_0"], os.makedev(235, 32))
        self.assertEqual(create.call_count, 3)
        self.assertFalse((dev / "xdma3_user").exists())
        (root / "module/xdma").rmdir()
        with self.assertRaisesRegex(RuntimeError, "load XDMA on the host"):
            local_runtime.device_numbers(self.db, root)

    def test_multi_card_database_is_rejected(self):
        self.db.write_text(json.dumps([{"bdf": "3f:00.0", "uid": "one"}, {"bdf": "af:00.0", "uid": "two"}]))
        with self.assertRaisesRegex(RuntimeError, "exactly one card"):
            local_runtime.selected_device(self.db)

    def test_launch_separates_programming_ssh_from_private_container_runtime(self):
        source, _, _ = self.hardware()
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        env = dict(os.environ, STAGE2_ARTIFACT_DIR=str(artifacts), FIRESIM_DB=str(self.db),
                   FIRESIM_HWDB_PATH=str(source), STAGE2_STATE_ROOT=str(self.root / "state"), RUN_ID="fixture",
                   FIRESIM_SSH_HOST="host.docker.internal", SSH_AUTH_SOCK=str(self.agent))
        for key in ("STAGE2_IMAGE", "STAGE2_DOCKER_HOST"):
            env.pop(key, None)
        result = subprocess.run(["bash", str(REPO / "scripts/run-stage2-local-docker.sh"),
                                 "--workload=squeezenet-scalar-4core", "--print-command"],
                                env=env, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = json.loads((self.root / "state/runs/fixture/launch-command.json").read_text())
        self.assertNotIn("--privileged", command)
        self.assertIn("--device-cgroup-rule=c *:* rwm", command)
        self.assertIn("--userns=host", command)
        self.assertNotIn("--pid=host", command)
        self.assertNotIn("--network=host", command)
        self.assertIn("pytorch-chipyard:stage2", command)
        mounts = [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--mount"]
        self.assertIn("type=bind,src=/sys,dst=/sys,readonly", mounts)
        self.assertIn(f"type=bind,src={self.db},dst=/run/stage2/fpga-db.json,readonly", mounts)
        self.assertIn("STAGE2_PROGRAM_FPGA_DB=" + str(self.db), command)
        self.assertFalse(any("/dev/bus/usb" in m for m in mounts))
        self.assertTrue(any("dst=/run/stage2/program-agent.sock,readonly" in m for m in mounts))
        self.assertFalse(any("src=/dev,dst=" in item or "Vivado" in item or "chipyard/scripts" in item for item in mounts))
        self.assertFalse(any("FIRESIM_SSH_HOST" in arg or "SSH_AUTH_SOCK" in arg for arg in command))

    def test_launch_requires_an_explicit_absolute_firesim_db(self):
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        for value in (None, "fpga.json"):
            env = dict(os.environ, STAGE2_ARTIFACT_DIR=str(artifacts), SSH_AUTH_SOCK=str(self.agent))
            env.pop("FIRESIM_DB", None)
            if value is not None:
                env["FIRESIM_DB"] = value
            result = subprocess.run([sys.executable, str(SUPPORT / "local-launch.py"), "--print-command"],
                                    env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FIRESIM_DB to the absolute path", result.stderr)

    def test_sudo_docker_keeps_the_selected_engine_caller_and_stop_command(self):
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        endpoint = "unix:///run/system-docker-fixture.sock"
        prefix = ["sudo", "-n", "/usr/bin/docker", "--host", endpoint]
        image_id = "sha256:" + "a" * 64
        metadata = [{"Id": image_id, "Config": {"Labels": {
            "org.pytorch-chipyard.stage2.execution": "full-container-u250-v1",
            "org.pytorch-chipyard.stage2.programmer": "host-vivado-ssh-v1"}}}]
        env = {"STAGE2_DOCKER_SUDO": "1", "STAGE2_DOCKER_HOST": endpoint,
               "DOCKER_HOST": "unix:///wrong-rootless.sock", "DOCKER_CONTEXT": "wrong-context",
               "STAGE2_ARTIFACT_DIR": str(artifacts), "FIRESIM_DB": str(self.db),
               "STAGE2_STATE_ROOT": str(self.root / "state"), "RUN_ID": "sudo-fixture",
               "SSH_AUTH_SOCK": str(self.agent)}
        handlers = {}

        def run(command, **kwargs):
            if command[len(prefix)] == "run":
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryFile() as lock, patch.dict(os.environ, env, clear=True), \
                patch.object(sys, "argv", ["launcher", "--preflight-only"]), \
                patch.object(launcher.os, "open", return_value=lock.fileno()), \
                patch.object(launcher, "require_no_host_simulation"), \
                patch.object(launcher, "device_numbers"), \
                patch.object(launcher.signal, "signal", side_effect=lambda sig, fn: handlers.update({sig: fn})), \
                patch.object(launcher.subprocess, "check_output", side_effect=['[]', json.dumps(metadata)]) as check, \
                patch.object(launcher.subprocess, "run", side_effect=run) as execute:
            launcher.main()
        calls = check.call_args_list + execute.call_args_list
        self.assertEqual([call.args[0][len(prefix)] for call in calls], ["info", "image", "run", "stop"])
        for call in calls:
            self.assertEqual(call.args[0][:len(prefix)], prefix)
            self.assertEqual(call.kwargs["env"]["DOCKER_HOST"], endpoint)
            self.assertNotIn("DOCKER_CONTEXT", call.kwargs["env"])
        command = execute.call_args_list[0].args[0]
        self.assertIn(image_id, command)
        self.assertIn("STAGE2_OUTPUT_UID=" + str(os.getuid()), command)
        self.assertIn("STAGE2_OUTPUT_GID=" + str(os.getgid()), command)

    def test_container_result_collection_marks_complete_only_after_guest_copy(self):
        destination = self.root / "results"
        slot = self.root / "runfarm/sim_slot_0"
        slot.mkdir(parents=True)
        (slot / "uartlog").write_text("UART fixture")
        job = types.SimpleNamespace(outputs=["/root/model.log"], simoutputs=["uartlog"])
        complete = []
        server = types.SimpleNamespace(get_job=lambda: job,
            get_local_job_results_dir_path=lambda: str(destination),
            get_host_instance=lambda: types.SimpleNamespace(get_sim_dir=lambda: str(slot.parent)),
            get_rootfs_name=lambda: "rootfs.img", write_job_complete_file=lambda: complete.append(True))
        with patch("guestfs_backend.Image") as image:
            image.return_value.__enter__.return_value.copy_out.side_effect = RuntimeError("missing guest result")
            with self.assertRaisesRegex(RuntimeError, "missing guest result"):
                local_runtime.copy_results(server, 0)
            self.assertEqual(complete, [])
            image.return_value.__enter__.return_value.copy_out.side_effect = None
            local_runtime.copy_results(server, 0)
            self.assertEqual(complete, [True])
            self.assertEqual((destination / "uartlog").read_text(), "UART fixture")

    def test_snapshot_patch_keeps_xdma_loaded_and_uses_guestfish(self):
        import ast
        root = self.root / "snapshot"
        fs = root / "sims/firesim"
        (fs / "deploy/runtools").mkdir(parents=True)
        (fs / "deploy/sudo-scripts").mkdir()
        topology = fs / "deploy/runtools/firesim_topology_elements.py"
        topology.write_text('class Server:\n    def copy(self, slotno):\n'
                            '        assert self.has_assigned_host_instance(), "copy requires assigned host instance"\n')
        manager = fs / "deploy/firesim"
        manager.write_text('def main():\n    if True:\n        env.abort_on_prompts = True\n'
                           '        vivado = os.environ["FIRESIM_VIVADO_DIR"]\n'
                           '        env.shell = "/usr/bin/env " + shlex.quote("XILINX_VIVADO=" + vivado) + " " + shlex.quote("PATH=" + vivado + "/bin:/usr/local/bin:/usr/bin:/bin") + " /bin/bash -c"\n')
        deploy = fs / "deploy/runtools/run_farm_deploy_managers.py"
        deploy.write_text('def flash():\n    if True:\n        if True:\n            for slotno in []:\n'
                          '                self.instance_logger(f"""Copying FPGA flashing scripts for {slotno}""")\n')
        snapshot.patch(root)
        ast.parse(topology.read_text())
        ast.parse(manager.read_text())
        ast.parse(deploy.read_text())
        self.assertIn("program(Path(bit)", deploy.read_text())
        self.assertNotIn("FIRESIM_VIVADO_DIR", manager.read_text())
        self.assertIn("return copy_results(self, slotno)", topology.read_text())
        for name in ("firesim-load-xdma-module", "firesim-remove-xdma-module"):
            helper = (fs / "deploy/sudo-scripts" / name).read_text()
            for prohibited in ("insmod", "rmmod", "modprobe"):
                self.assertNotIn(prohibited, helper)
            self.assertIn("test -d /sys/module/xdma", helper)

    def test_sudo_profile_drops_caller_environment_and_pins_smoke_execution(self):
        import pwd
        account = pwd.getpwuid(os.getuid())
        config = {"user": account.pw_name, "image_id": "sha256:" + "a" * 64,
                  "smoke_inputs": "/usr/local/share/fixed-smoke", "fpga_db": "/opt/firesim-db0.json", "ssh_host": "host.docker.internal", "vivado": "/host/vivado"}
        args = types.SimpleNamespace(run_id=None, resume="fixture", preflight_only=False, ssh_agent=str(self.agent))
        with patch.dict(os.environ, DOCKER_HOST="unix:///attacker.sock", DOCKER_CONFIG="/tmp/attacker",
                        STAGE2_IMAGE="attacker:image", FIRESIM_HWDB_PATH="/tmp/native-driver.yaml",
                        FIRESIM_DB="/tmp/other-card.json",
                        SUDO_UID="99999", PYTHONPATH="/tmp/attacker", STAGE2_STATE_ROOT="/tmp/attacker"):
            command, env = privileged.invocation(config, args, account.pw_uid)
        self.assertEqual(env["FIRESIM_DB"], config["fpga_db"])
        self.assertEqual(env["STAGE2_IMAGE"], config["image_id"])
        self.assertEqual(env["STAGE2_STATE_ROOT"], "/var/lib/pytorch-chipyard-stage2-local")
        self.assertEqual(env["STAGE2_DOCKER_HOST"], "unix:///var/run/docker.sock")
        for key in ("SUDO_UID", "SUDO_GID", "PYTHONPATH", "FIRESIM_HWDB_PATH", "DOCKER_HOST"):
            self.assertNotIn(key, env)
        self.assertIn("--smoke-test", command)
        self.assertIn("--resume", command)
        self.assertIn("-I", command)
        with self.assertRaisesRegex(RuntimeError, "installed for"):
            privileged.invocation(config, args, account.pw_uid + 1)
        args.resume = "../../escape"
        with self.assertRaisesRegex(RuntimeError, "invalid RUN_ID"):
            privileged.invocation(config, args, account.pw_uid)


if __name__ == "__main__":
    unittest.main()
