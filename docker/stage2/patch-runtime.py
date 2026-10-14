#!/usr/bin/env python3
"""Apply container-only adaptations to the pinned runtime sources."""
import argparse
from pathlib import Path
import shutil


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"snapshot patch context mismatch: {path}: {old[:65]!r}")
    path.write_text(text.replace(old, new, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    root = args.snapshot.resolve()
    if root != Path("/opt/chipyard"):
        parser.error("patches may only be applied to /opt/chipyard inside the image")
    fm = root / "software/firemarshal"
    shutil.copy2(Path(__file__).with_name("guestfs_backend.py"), fm / "wlutil/stage2_guestfs.py")
    replace_once(fm / "wlutil/wlutil.py",
                 "    global pwdlessSudoCmd\n\n    assert imgPath.is_file()",
                 '    global pwdlessSudoCmd\n\n'
                 '    if os.environ.get("PYTORCH_CHIPYARD_STAGE2_CONTAINER") == "1":\n'
                 '        raise RuntimeError("Stage2 supports disk images via guestfish; local mounts/nodisk are disabled")\n\n'
                 '    assert imgPath.is_file()')
    replace_once(fm / "wlutil/wlutil.py",
                 "    assert direction in ['in', 'out'], f\"direction={direction} must be either 'in' or 'out'\"\n",
                 "    assert direction in ['in', 'out'], f\"direction={direction} must be either 'in' or 'out'\"\n"
                 '    if os.environ.get("PYTORCH_CHIPYARD_STAGE2_CONTAINER") == "1":\n'
                 '        from .stage2_guestfs import copy_image_files\n'
                 '        return copy_image_files(img, files, direction)\n')
    replace_once(fm / "wlutil/build.py", "def makeBin(config, nodisk=False):\n",
                 'def makeBin(config, nodisk=False):\n'
                 '    if nodisk and os.environ.get("PYTORCH_CHIPYARD_STAGE2_CONTAINER") == "1":\n'
                 '        raise RuntimeError("nodisk is outside the Stage2 container interface")\n')
    # Explicit SSH identity: container UID 0 is not the FPGA host's root user.
    replace_once(root / "sims/firesim/deploy/firesim", "    env.timeout = 100\n",
                 '    if os.environ.get("PYTORCH_CHIPYARD_STAGE2_CONTAINER") == "1":\n'
                 '        import shlex\n'
                 '        env.user = os.environ["FIRESIM_SSH_USER"]\n'
                 '        env.use_ssh_config = True\n'
                 '        env.ssh_config_path = os.environ["STAGE2_SSH_CONFIG"]\n'
                 '        env.abort_on_prompts = True\n'
                 '        vivado = os.environ["FIRESIM_VIVADO_DIR"]\n'
                 '        env.shell = "/usr/bin/env " + shlex.quote("XILINX_VIVADO=" + vivado) + " " + shlex.quote("PATH=" + vivado + "/bin:/usr/local/bin:/usr/bin:/bin") + " /bin/bash -c"\n'
                 '    env.timeout = 100\n')
    replace_once(root / "sims/firesim/deploy/firesim", "    env.disable_known_hosts = True\n",
                 '    env.disable_known_hosts = os.environ.get("PYTORCH_CHIPYARD_STAGE2_CONTAINER") != "1"\n'
                 '    if not env.disable_known_hosts:\n'
                 '        env.reject_unknown_hosts = True\n')
    # Buildroot uses the pinned fakeroot source installed during the image build.
    (fm / "boards/default/distros/br/buildroot/local.mk").write_text(
        "FAKEROOT_OVERRIDE_SRCDIR = /opt/chipyard/software/firemarshal/boards/default/distros/br/fakeroot-1.31\n")


if __name__ == "__main__":
    main()
