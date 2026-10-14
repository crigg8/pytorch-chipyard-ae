# Stage 2 Docker build and execution

The execution image contains FireMarshal, FireSim Manager, native simulation
drivers, Linux/OpenSBI/Buildroot sources, the runtime Conda environment, and a
libguestfs appliance. Vivado 2021.1, `hw_server`, and the XDMA kernel module stay
on the FPGA host. Stage 2 uses SSH only to request host FPGA programming; the
simulation runs inside Docker against the selected card's XDMA device nodes.

The successful reference smoke run is recorded in
[`docker/stage2/validation.json`](../docker/stage2/validation.json). It validates
three GEMM workloads on U250, not the complete paper evaluation or every image
rebuilt from this checkout.

## Build the image

```bash
docker build -f docker/stage2.Dockerfile -t pytorch-chipyard:stage2 .
docker run --rm --init --shm-size=1g pytorch-chipyard:stage2 --check-tools
```

Download the release bitstreams using `scripts/download-bitstreams.sh` first.
The build uses this checkout and the downloaded `bitstreams/` directory. It
fetches the pinned FireSim, FireMarshal, Linux, OpenSBI, Buildroot, BusyBox,
and guest driver sources and installs runtime packages inside `/opt/chipyard`.
It needs no host Chipyard installation, prepared context, cache environment
variable, or Stage 1 image. Docker manages its own build cache.

Source commits are in `docker/stage2/sources.lock`; runtime package versions
are in `conda-linux-64.lock` and `requirements.txt` in the same directory.
The checked-in patches preserve the AE runtime behavior, including the RVV
Linux changes. The image generates its bundled HWDB and input hashes during
the build. Host Vivado and XDMA are needed only when executing on the FPGA.

`--check-tools` checks tool startup, driver libraries, guest-image access,
and internal SSH without FPGA access. Hardware validation is a separate run.

## Transfer the image to system Docker

Rootless and system Docker have separate image stores. Export from the build
engine, using the same checkout that will be used for installation:

```bash
mkdir -p "$HOME/stage2-delivery"
python3 scripts/export-stage2-local-image.py \
  --image pytorch-chipyard:stage2 \
  --output "$HOME/stage2-delivery/stage2-local.tar" \
  --delivery "$HOME/stage2-delivery/delivery.json"
```

The export records the image ID, archive checksum, and hashes of the restricted
launcher files. The restricted-profile installer below verifies and loads it.
For a general system-Docker launch, an administrator can instead load it with:

```bash
sudo docker --host unix:///var/run/docker.sock image load \
  --input "$HOME/stage2-delivery/stage2-local.tar"
```

This administrator command is not a sudo rule to give to `ae`. Keep `ae`'s
existing restricted permissions when using the fixed smoke profile.

## FPGA host preparation

`FIRESIM_DB` selects the physical card. The bitstream/driver configuration
remains the separate `FIRESIM_HWDB_PATH`, which defaults to the image's bundled HWDB.

An administrator prepares these once on the physical execution host:

- A working Vivado **2021.1** installation with `bin/vivado` and `bin/hw_server`.
  Set `FIRESIM_VIVADO_DIR` to its absolute directory for a general launch, or
  pass `--vivado` during restricted-profile installation. No Vivado or Lab
  Edition files are copied into the image.
- The already-loaded XDMA module and a root-owned single-card database
  selected by `FIRESIM_DB`. The database must contain exactly the selected BDF
  and JTAG identity. The container refreshes only that card's device nodes
  after programming; it does not load or unload the host module.
- The existing root-owned FireSim helper
  `/usr/local/bin/firesim-fpga-util.py` and its companion `pcielib.py`, `util.py`,
  and `program_fpga.tcl`. The SSH account needs the existing noninteractive
  sudo permission for FPGA programming through that helper.
- An SSH server reachable from Docker, public host keys readable in `/etc/ssh`,
  and an agent key that logs into the same physical host as the chosen account.
  The host identity and selected FPGA are checked before programming.
- A shared execution lock at `/var/lock/firesim-stage2.lock`, owned by
  `root:firesim` with mode `0660`. Prepare it once, with no active FPGA job:

```bash
sudo touch /var/lock/firesim-stage2.lock
sudo chown root:firesim /var/lock/firesim-stage2.lock
sudo chmod 0660 /var/lock/firesim-stage2.lock
```

The launcher refuses concurrent FireSim/programming processes. It uses system
Docker with `--userns=host`, device cgroup permission, and device-node creation
capability; it does not use `--privileged`, a Docker socket mount, host home
mounts, or a host Vivado mount. Full-experiment invocation still requires
administrator-approved access to system Docker.

## Restricted ae smoke profile

Run the Stage 1 smoke test shown in the README first. It compiles and links
all three ELFs with the image's bundled GCC 12.2.0.
An **administrator** then runs this from the same source checkout used to
export the delivery:

```bash
sudo /usr/bin/python3 -I scripts/install-stage2-local-smoke-launcher.py \
  --delivery /absolute/path/to/stage2-delivery/delivery.json \
  --smoke-inputs /absolute/path/to/pytorch-chipyard-ae/results/smoke-test/artifacts/pytorch-chipyard \
  --user ae \
  --firesim-db /absolute/path/to/firesim-db.json \
  --ssh-host fpga.example.org \
  --vivado /absolute/path/to/Vivado/2021.1
sudo visudo -c
sudo -l -U ae
```

Replace the example SSH host and paths with the prepared host's values. The
installer pins the image ID, copies the inputs and launcher into protected
root-owned locations, and adds only this new sudo entry:

```sudoers
ae ALL=(root) NOPASSWD: NOSETENV: /usr/local/sbin/run-stage2-local-smoke
```

It does not grant general Docker or shell access. The command accepts an
`ae`-owned SSH agent socket, a run ID or resume ID, and `--preflight-only`;
it does not accept arbitrary images, input paths, HWDBs, or Docker options.
Existing FireSim helper permissions remain an administrator-managed prerequisite.

In the **ae terminal**:

```bash
if [[ ! -S "${SSH_AUTH_SOCK:-}" ]]; then eval "$(ssh-agent -s)"; fi
if ! ssh-add -l >/dev/null 2>&1; then ssh-add; fi
bash scripts/run-smoke-test-stage2-docker.sh --ssh-agent "$SSH_AUTH_SOCK"
```

To resume the same installed smoke run:

```bash
RUN_ID=previous-run-id bash scripts/run-smoke-test-stage2-docker.sh \
  --ssh-agent "$SSH_AUTH_SOCK" --resume
```

Its results are under
`/var/lib/pytorch-chipyard-stage2-local/runs/<run-id>/results`.
The public command selects the restricted launcher when the account cannot
access system Docker. It checks that the image in the build Docker engine
and the current Stage 1 outputs match the installed profile before running.
Changed images or inputs require a profile refresh; the command refuses to
silently test older installed artifacts.
Reinstalling a profile changes its pinned configuration; finish or preserve
ongoing runs before an administrator updates the installation.

## Simple experiment inputs

`--experiment=simple` requires `artifact-gemmini-max-autotune-simple/gemmini`
in addition to the shared SqueezeNet/OPT inputs. A full Stage 1 run produces
the full Figure 11 artifact instead. If the reduced artifact is missing,
generate it in Stage 1 before running `prepare-stage2-inputs.sh`:

```bash
docker run --rm -it \
  -e PYTORCH_CHIPYARD_SIMPLE_STAGE2=1 \
  -v "$PWD/examples:/opt/pytorch-chipyard/examples" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  pytorch-chipyard:stage1 bash scripts/run_gemmini_autotune.sh
```

Alternatively, for an evaluation restricted to the simple experiment, pass
`-e PYTORCH_CHIPYARD_SIMPLE_STAGE2=1` to the README's Stage 1 Docker command.
That prepares the reduced workload set directly. Full-model experiment
commands still require the corresponding full Stage 1 inputs.

## Full experiments and results

The public entrypoint `scripts/run-stage2-docker.sh` selects the full container
launcher. The legacy `scripts/run-stage2.sh` is not the host entrypoint for this
workflow. Follow the README's Stage 1 ELF-linking and experiment commands.

The general launcher uses these settings:

| Setting | Meaning |
| --- | --- |
| `STAGE2_IMAGE` | Final image tag/ID; default `pytorch-chipyard:stage2` |
| `STAGE2_ARTIFACT_DIR` | Completed full-model ELF/input tree; default `<checkout>/examples` |
| `STAGE2_SMOKE_ARTIFACT_DIR` | Smoke input tree; default `<checkout>/results/smoke-test/artifacts/pytorch-chipyard` |
| `FIRESIM_SSH_HOST`, `FIRESIM_SSH_USER` | Host programming SSH destination and account |
| `FIRESIM_VIVADO_DIR` | Absolute host Vivado 2021.1 directory; set explicitly |
| `FIRESIM_DB` | Required absolute path to the physical FPGA database (one card, with PCIe BDF and JTAG UID) |
| `FIRESIM_HWDB_PATH` | Optional external local HWDB; default uses the image's bundled HWDB |
| `STAGE2_STATE_ROOT` | Results/checkpoints root; default `~/.local/share/pytorch-chipyard-stage2-local` |
| `STAGE2_BUILD_JOBS` | Guest build parallelism; default `8` |
| `RUN_ID` | Set to the printed previous ID when using `--resume` |

An external HWDB may reference local bitstream/driver archives; the launcher
mounts only those files read-only and rewrites the paths for the container.
HWDB archives, input files, the image ID, and programming settings are hashed
into each run's identity. A changed identity cannot silently resume an old run.

`--preflight-only` does not program the FPGA or run the simulator, but it does
check SSH and open/close the selected device nodes. It creates a run checkpoint;
continue with its printed `RUN_ID` and `--resume`. Successful workloads retain
model/autotune logs, UART, output binaries where applicable, and configuration
metadata. Failed runs preserve diagnostic logs and images for investigation.

TVM/Verilator and Table 4 wall-clock measurements are excluded. The paper
plotter can use `reference-results/table4.csv`; do not describe that reference
as a fresh Docker measurement. Full-model experiments need their own FPGA
validation after the three-workload smoke test.
