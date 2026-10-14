#!/usr/bin/env python3
"""Check the complete userspace without exposing an FPGA to the container."""
import os
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch


def check_result_copy(env):
    """Exercise result extraction from a real ext4 image without loop mounts."""
    from guestfs_backend import Image
    from local_runtime import copy_results
    with tempfile.TemporaryDirectory(prefix="stage2-results-check-") as tmp:
        root = Path(tmp)
        slot = root / "runfarm/sim_slot_0"
        slot.mkdir(parents=True)
        disk = slot / "rootfs.img"
        with disk.open("wb") as stream:
            stream.truncate(64 * 1024 * 1024)
        subprocess.run(["/usr/sbin/mkfs.ext4", "-q", "-F", str(disk)], env=env, check=True)
        payloads = {"model.log": b"model result\n", "autotune.log": b"autotune result\n", "output.bin": bytes(range(256))}
        with Image(disk) as image:
            image.command("mkdir-p", "/root/results")
            for name, payload in payloads.items():
                source = root / name
                source.write_bytes(payload)
                image.copy_in(source, "/root/results/" + name)
        (slot / "uartlog").write_text("simulation completed\n")
        destination = root / "results"
        marker = root / "complete"
        server = SimpleNamespace(
            get_job=lambda: SimpleNamespace(outputs=["/root/results/" + n for n in payloads], simoutputs=["uartlog"]),
            get_local_job_results_dir_path=lambda: str(destination),
            get_host_instance=lambda: SimpleNamespace(get_sim_dir=lambda: str(slot.parent)),
            get_rootfs_name=lambda: "rootfs.img", write_job_complete_file=lambda: marker.touch())
        copy_results(server, 0)
        for name, payload in payloads.items():
            assert (destination / name).read_bytes() == payload, name
        assert (destination / "uartlog").read_text() == "simulation completed\n"
        assert marker.exists()
    print("STAGE2_LOCAL_RESULT_COPY_STATUS=PASS", flush=True)


def check_internal_ssh(prefix, env):
    """Exercise the same OpenSSH + Fabric path without an FPGA or host login."""
    # sshd StrictModes rejects authorized keys below world-writable /tmp.
    with tempfile.TemporaryDirectory(prefix="stage2-ssh-check-", dir="/root/.ssh") as tmp:
        root = Path(tmp)
        key = root / "key"
        subprocess.run(["ssh-keygen", "-q", "-t", "rsa", "-b", "2048", "-N", "", "-f", str(key)], check=True)
        (root / "known_hosts").write_text("[127.0.0.1]:2222 " + (root / "key.pub").read_text())
        config = root / "sshd_config"
        config.write_text(f"Port 2222\nListenAddress 127.0.0.1\nHostKey {key}\nPidFile {root}/pid\n"
                          f"AuthorizedKeysFile {key}.pub\nPermitRootLogin prohibit-password\n"
                          "PasswordAuthentication no\nUsePAM no\nAllowUsers root\nSubsystem sftp internal-sftp\n")
        client = root / "ssh_config"
        client.write_text(f"Host 127.0.0.1 stage2-programmer\n  HostName 127.0.0.1\n  User root\n  Port 2222\n  IdentityFile {key}\n"
                          f"  UserKnownHostsFile {root}/known_hosts\n  StrictHostKeyChecking yes\n  BatchMode yes\n")
        with (root / "sshd.log").open("w+") as log:
            server = subprocess.Popen(["/usr/sbin/sshd", "-D", "-e", "-f", str(config)], env=env, stdout=log, stderr=log)
            try:
                for _ in range(50):
                    checked = subprocess.run(["ssh", "-F", str(client), "127.0.0.1", "true"], env=env,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if checked.returncode == 0:
                        break
                    if server.poll() is not None:
                        break
                    time.sleep(0.1)
                else:
                    log.seek(0)
                    raise RuntimeError("container SSH startup timed out: " + log.read())
                if checked.returncode:
                    log.seek(0)
                    raise RuntimeError(log.read())
                program = ('from fabric.api import env, run\n'
                           'import sys\n'
                           'env.host_string="127.0.0.1"; env.user="root"; env.port=2222\n'
                           'env.key_filename=sys.argv[1]; env.use_ssh_config=True; env.ssh_config_path=sys.argv[2]\n'
                           'env.system_known_hosts=sys.argv[3]\n'
                           'env.abort_on_prompts=True; env.disable_known_hosts=False; env.reject_unknown_hosts=True\n'
                           'env.shell="/usr/bin/env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /bin/bash -c"\n'
                           'run("command -v lsmod; command -v setpci; command -v screen")\n'
                           'assert str(run("id -u")).strip()=="0"\n')
                subprocess.run([str(prefix / ".conda-env/bin/python3"), "-c", program,
                                str(key), str(client), str(root / "known_hosts")], env=env, check=True)
                check_programming_transport(root, client)
            finally:
                server.terminate()
                server.wait(timeout=10)


def check_programming_transport(root, client):
    """Real SSH binary transfer and receiver hash checks, without FPGA access."""
    import host_programmer
    original = host_programmer.receiver_source()
    # Load the real receiver functions without calling hardware validation.
    fixture = ("namespace={'__name__':'fixture'}\nexec(" + repr(original) + ", namespace)\n"
               "import json, sys, tempfile\nfrom pathlib import Path\n"
               "config=json.loads(sys.argv[1])\n"
               "if 'bitstream_sha256' not in config:\n"
               "    print(json.dumps({'fixture': 'passed'}))\n"
               "else:\n"
               "    with tempfile.TemporaryDirectory() as tmp:\n"
               "        namespace['receive'](sys.stdin.buffer, Path(tmp)/'firesim.bit', "
               "config['bitstream_bytes'], config['bitstream_sha256'])\n")
    payload = root / "payload.bit"
    payload.write_bytes(bytes(range(256)) * 4096)
    config = {"bitstream_bytes": payload.stat().st_size,
              "bitstream_sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
    with patch.dict(os.environ, STAGE2_PROGRAM_SSH_CONFIG=str(client)), \
            patch.object(host_programmer, "receiver_source", return_value=fixture):
        assert host_programmer.request({}) == {"fixture": "passed"}
        host_programmer.request(config, payload)
        config["bitstream_sha256"] = "0" * 64
        try:
            host_programmer.request(config, payload)
        except subprocess.CalledProcessError:
            pass
        else:
            raise RuntimeError("host receiver accepted a corrupt bitstream")
    print("STAGE2_LOCAL_HOST_SSH_TRANSFER_STATUS=PASS", flush=True)


def check_native_drivers(env):
    """Resolve the packaged drivers' native libraries without starting an FPGA."""
    archives = sorted(Path("/opt/firesim-artifacts").glob("*/driver-bundle.tar.gz"))
    if not archives:
        raise RuntimeError("no packaged FireSim drivers")
    for archive in archives:
        with tempfile.TemporaryDirectory(prefix="stage2-driver-check-") as tmp:
            subprocess.run(["tar", "xzf", str(archive), "-C", tmp], env=env, check=True)
            driver = Path(tmp) / "FireSim-xilinx_alveo_u250"
            checked = subprocess.run(["/usr/bin/ldd", "-r", str(driver)], env=env,
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if checked.returncode or "not found" in checked.stdout or "undefined symbol" in checked.stdout:
                raise RuntimeError("unusable native driver " + str(archive) + ":\n" + checked.stdout)
    print("STAGE2_LOCAL_NATIVE_DRIVERS_STATUS=PASS count=" + str(len(archives)), flush=True)


def check_compilers(prefix, env):
    """Compile and link on the freshly installed host and RISC-V toolchains."""
    subprocess.run([str(prefix / ".conda-env/bin/python3"), "-m", "pip", "check"], env=env, check=True)
    with tempfile.TemporaryDirectory(prefix="stage2-compiler-check-") as tmp:
        root = Path(tmp)
        source = root / "hello.c"
        source.write_text('#include <stdio.h>\nint main(void) { puts("compiler check"); return 0; }\n')
        host = root / "host"
        guest = root / "guest.elf"
        subprocess.run(["gcc", str(source), "-o", str(host)], env=env, check=True)
        subprocess.run([str(host)], env=env, check=True)
        subprocess.run(["riscv64-unknown-linux-gnu-gcc", str(source), "-fopenmp", "-o", str(guest)], env=env, check=True)
        header = subprocess.check_output(["readelf", "-h", str(guest)], env=env, text=True)
        assert "RISC-V" in header and "double-float ABI" in header
    subprocess.run(["bash", "-c", "source ./sourceme-manager.sh --skip-ssh-setup; test \"$FIRESIM_ENV_SOURCED\" = 1"],
                   cwd=prefix / "sims/firesim", env=env, check=True)
    print("STAGE2_COMPILERS_STATUS=PASS", flush=True)


def main():
    prefix = Path("/opt/chipyard")
    env = dict(os.environ)
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    subprocess.run([str(prefix / ".conda-env/bin/python3"),
                    str(Path(__file__).with_name("check-firesim-config.py"))], env=env, check=True)
    check_compilers(prefix, env)
    check_internal_ssh(prefix, env)
    subprocess.run(["/usr/bin/guestfish", "--version"], env=env, check=True)
    check_result_copy(env)
    check_native_drivers(env)
    subprocess.run([str(prefix / ".conda-env/bin/python3"),
                    str(prefix / "sims/firesim/deploy/firesim"), "--help"], env=env, check=True)
    subprocess.run([str(prefix / ".conda-env/bin/python3"),
                    str(prefix / "software/firemarshal/marshal"), "--help"],
                   cwd=prefix / "software/firemarshal", env=env, check=True)
    assert not Path("/opt/Xilinx").exists(), "vendor tools must remain on the host"
    print("STAGE2_LOCAL_TOOLS_STATUS=PASS")


if __name__ == "__main__":
    main()
