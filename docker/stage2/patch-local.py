#!/usr/bin/env python3
"""Adapt an already prepared Stage2 image to run its run farm locally."""
import argparse
from pathlib import Path
import shutil


def replace_once(path, old, new):
    content = path.read_text()
    if content.count(old) != 1:
        raise RuntimeError("unexpected FireSim snapshot: " + str(path))
    path.write_text(content.replace(old, new, 1))


def patch(root):
    fs = root / "sims/firesim"
    topology = fs / "deploy/runtools/firesim_topology_elements.py"
    replace_once(topology, '        assert self.has_assigned_host_instance(), "copy requires assigned host instance"\n',
                 '        import os\n'
                 '        if os.environ.get("STAGE2_EXECUTION") == "local":\n'
                 '            sys.path.insert(0, "/opt/stage2-support")\n'
                 '            from local_runtime import copy_results\n'
                 '            return copy_results(self, slotno)\n'
                 '        assert self.has_assigned_host_instance(), "copy requires assigned host instance"\n')
    # Fabric's SSH sessions need the same container-only settings as the manager.
    replace_once(fs / "deploy/firesim", '        env.abort_on_prompts = True\n',
                 '        env.abort_on_prompts = True\n'
                 '        if os.environ.get("STAGE2_EXECUTION") == "local":\n'
                 '            env.key_filename = "/root/.ssh/id_rsa"\n'
                 '            env.forward_agent = False\n')
    replace_once(fs / "deploy/firesim", '        vivado = os.environ["FIRESIM_VIVADO_DIR"]\n', '')
    replace_once(fs / "deploy/firesim",
                 '        env.shell = "/usr/bin/env " + shlex.quote("XILINX_VIVADO=" + vivado) + " " + shlex.quote("PATH=" + vivado + "/bin:/usr/local/bin:/usr/bin:/bin") + " /bin/bash -c"\n',
                 '        env.shell = "/usr/bin/env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /bin/bash -c"\n')
    deploy = fs / "deploy/runtools/run_farm_deploy_managers.py"
    replace_once(deploy, '                self.instance_logger(f"""Copying FPGA flashing scripts for {slotno}""")\n',
                 '                if os.environ.get("STAGE2_EXECUTION") == "local":\n'
                 '                    import sys\n'
                 '                    sys.path.insert(0, "/opt/stage2-support")\n'
                 '                    from host_programmer import program\n'
                 '                    program(Path(bit), self.slot_to_bdf(slotno, self.parent_node.get_fpga_db()))\n'
                 '                    continue\n\n'
                 '                self.instance_logger(f"""Copying FPGA flashing scripts for {slotno}""")\n')
    helpers = fs / "deploy/sudo-scripts"
    for name in ("firesim-load-xdma-module", "firesim-remove-xdma-module"):
        (helpers / name).write_text('#!/bin/bash\nset -eu\n'
            'test -d /sys/module/xdma || { echo "XDMA must already be loaded on the host" >&2; exit 1; }\n'
            'echo "[stage2-local] keeping host XDMA module loaded"\n')
    (helpers / "firesim-chmod-xdma-perm").write_text(
        '#!/bin/bash\nset -eu\nexec /usr/bin/python3 /opt/stage2-support/local_runtime.py devices\n')
    (helpers / "firesim-change-pcie-perms").write_text(
        '#!/bin/bash\nset -eu\nexec /usr/bin/python3 /opt/stage2-support/local_runtime.py pci "$@"\n')
    return fs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    root = args.snapshot.resolve()
    if root != Path("/opt/chipyard") or Path(__file__).resolve() != Path("/opt/stage2-support/patch-local.py"):
        parser.error("patches may only be applied inside the Stage2 image")
    fs = patch(root)
    # These helpers belong to the image; no host /usr/local/bin is mounted.
    for path in (fs / "deploy/sudo-scripts").glob("firesim-*"):
        shutil.copy2(path, Path("/usr/local/bin") / path.name)
    for path in (fs / "platforms/xilinx_alveo_u250/scripts").iterdir():
        if path.suffix in (".py", ".tcl"):
            shutil.copy2(path, Path("/usr/local/bin") / path.name)


if __name__ == "__main__":
    main()
