"""Disk-image file operations using a private, unprivileged guestfish appliance.

Only offline raw ext4 images are supported. No host filesystem mount is used.
Runtime integration tests are deliberately deferred until FPGA/AE scheduling.
"""
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import tempfile


def guest_path(path):
    path = PurePosixPath(str(path))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"expected absolute guest path without '..': {path}")
    return str(path)


class Image:
    def __init__(self, path, readonly=False):
        self.path = str(Path(path).resolve(strict=True))
        self.readonly = readonly
        self.pid = None
        self.env = dict(os.environ, LIBGUESTFS_BACKEND="direct",
                        LIBGUESTFS_BACKEND_SETTINGS="force_tcg")
        self.env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        self.env["LIBGUESTFS_HV"] = "/usr/bin/qemu-system-x86_64"
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"):
            self.env.pop(name, None)

    def command(self, *args):
        return subprocess.check_output(
            ["/usr/bin/guestfish", f"--remote={self.pid}", *map(str, args)],
            text=True, env=self.env).rstrip("\n")

    def __enter__(self):
        reply = subprocess.check_output(["/usr/bin/guestfish", "--listen"],
                                        text=True, env=self.env)
        match = re.search(r"GUESTFISH_PID=(\d+)", reply)
        if not match:
            raise RuntimeError(f"guestfish did not return a server PID: {reply!r}")
        self.pid = match.group(1)
        try:
            self.command("add-drive-opts", self.path, "format:raw",
                         "readonly:" + str(self.readonly).lower())
            self.command("run")
            self.command("mount-ro" if self.readonly else "mount", "/dev/sda", "/")
        except BaseException:
            self.__exit__(*__import__("sys").exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        failure = None
        for args in [("umount-all",), ("sync",), ("exit",)]:
            try:
                self.command(*args)
            except subprocess.CalledProcessError as error:
                failure = failure or error
        if failure is not None and exc_type is None:
            raise failure

    def is_dir(self, path):
        return self.command("is-dir", guest_path(path), "followsymlinks:true") == "true"

    def copy_in(self, source, destination):
        source = Path(source).absolute()
        if not source.exists() and not source.is_symlink():
            raise FileNotFoundError(source)
        destination = PurePosixPath(guest_path(destination))
        target = destination / source.name if self.is_dir(destination) else destination
        if str(target) == "/":
            raise ValueError("copy-in must name a file or directory beneath /")
        if not self.is_dir(target.parent):
            raise FileNotFoundError(f"guest parent directory missing: {target.parent}")
        # Preserve an existing destination directory's permissions, as the
        # upstream mount/cp implementation does after merging an overlay.
        old_mode = None
        if self.is_dir(target):
            values = self.command("statns", target)
            match = re.search(r"st_mode:\s*(\d+)", values)
            if match:
                old_mode = stat.S_IMODE(int(match.group(1)))
        with tempfile.TemporaryDirectory(prefix="stage2-copy-in-") as temporary:
            archive = Path(temporary) / "files.tar"
            with tarfile.open(archive, "w", dereference=False) as tar:
                tar.add(source, arcname=str(target.relative_to("/")))
            self.command("tar-in", archive, "/")
        if old_mode is not None:
            self.command("chmod", f"0{old_mode:o}", target)

    def copy_out(self, source, destination):
        source = guest_path(source)
        if source == "/":
            raise ValueError("export individual files/directories, not the whole rootfs")
        destination = Path(destination).absolute()
        with tempfile.TemporaryDirectory(prefix="stage2-copy-out-") as temporary:
            self.command("copy-out", source, temporary)
            exported = Path(temporary) / PurePosixPath(source).name
            subprocess.run(["cp", "-a", "-f", str(exported), str(destination)], check=True)


def copy_image_files(image, files, direction):
    if direction not in ("in", "out"):
        raise ValueError(direction)
    with Image(image, readonly=direction == "out") as filesystem:
        for item in files:
            if direction == "in":
                filesystem.copy_in(item.src, item.dst)
            else:
                filesystem.copy_out(item.src, item.dst)
