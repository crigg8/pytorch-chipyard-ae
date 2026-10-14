# PyTorch-Chipyard Artifact

A reusable compiler-stack foundation for ML research on Chipyard hardware.
This README provides the end-to-end commands for reproducing the reported
results on a FireSim-enabled local FPGA server.

## Software versions

- Vivado 2021.1
- XRT 2.16.204
- Verilator 5.022
- RISC-V GNU Toolchain 12.2.0
- Gemmini gemmini-microtvm (3b07a14b)

## Evaluated FPGA Architecture

- AMD/Xilinx Alveo U250 FPGA

## Evaluated Torch Models

- ResNet50
- MobilenetV2
- AlexNet
- SqueezeNet
- OPT-125m
- GPT2-124m
- GPT-Neo-125m
- Pythia-160m

## 1. Unpack and build the Docker images

Unpack the Zenodo archive in the home directory so that the artifact root is
`~/pytorch-chipyard-ae`, then enter that directory:

Because the Docker image build and Stage 2 simulations can take a long time, run them inside a `tmux` session so that they continue if the SSH connection is interrupted.

```bash
# or Download it from Zenodo
git clone --branch v1.0.0 --depth 1 \
  https://github.com/crigg8/pytorch-chipyard-ae.git \
  pytorch-chipyard-ae
cd ~/pytorch-chipyard-ae
bash scripts/download-bitstreams.sh

source ~/.bashrc

# Stage 1
docker build \
  -f docker/stage1.Dockerfile \
  -t pytorch-chipyard:stage1 \
  .

# Stage 2
docker build \
  -f docker/stage2.Dockerfile \
  -t pytorch-chipyard:stage2 \
  .
```

## 2. Run the bounded smoke test

### Stage 1

Compile and link the three smoke GEMMs inside the Stage 1 image:

```bash
mkdir -p results

docker run --rm -it \
  -v "$PWD/results:/opt/pytorch-chipyard/results" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  pytorch-chipyard:stage1 \
  bash scripts/run-smoke-test-stage1.sh
```

A successful run prints `SMOKE_STAGE1_STATUS=PASS`.

### Stage 2

Run the FPGA tests with system Docker access and the host login key loaded
in your SSH agent:

```bash
export FIRESIM_SSH_HOST=fpga.example.org  # replace with the FPGA host IP/DNS
export FIRESIM_SSH_USER="$(id -un)"
export FIRESIM_VIVADO_DIR=/absolute/path/to/Vivado/2021.1
export FIRESIM_DB=/absolute/path/to/firesim-db.json

bash scripts/run-smoke-test-stage2-docker.sh --ssh-agent "$SSH_AUTH_SOCK"
```

Runs **RVV 4-core, Gemmini 4-core, and scalar 16-core** sequentially.
A successful run prints `SMOKE_TEST_STATUS=PASS`.

## 3. Run Stage 1

```bash
mkdir -p examples results

docker run --rm -it \
  -v "$PWD/examples:/opt/pytorch-chipyard/examples" \
  -v "$PWD/results:/opt/pytorch-chipyard/results" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  pytorch-chipyard:stage1 \
  bash scripts/run-stage1.sh
```

Stage 1 compiles the paper workloads and records PyTorch-Chipyard's Docker-side compile measurement.
A successful run prints `STAGE1_STATUS=PASS`.

## 4. Run Stage 2 in Docker

First link the Stage 1 outputs using GCC 12.2.0 included in the Stage 1 image. For `--experiment=simple`, prepare its reduced
Figure 11 input [before linking](docs/stage2-docker.md#simple-experiment-inputs).

```bash
bash scripts/prepare-stage2-inputs.sh
```

Stage 2 mounts those inputs read-only, builds FireMarshal images, runs
FireSim Manager and the native simulation driver inside the container, and
collects results into a separate run directory. Only FPGA programming goes
over SSH to host Vivado; the host XDMA module remains loaded.

Use system Docker and the SSH/Vivado settings from the smoke test.

```bash
export STAGE2_ARTIFACT_DIR="$PWD/examples"
export STAGE2_STATE_ROOT="$HOME/.local/share/pytorch-chipyard-stage2-local"

# Choose one experiment per invocation; each new invocation creates a run ID.
# Simple test: partial Figures 6ab, 8a, 9, 11, 13ac
bash scripts/run-stage2-docker.sh --experiment=simple

# Figures 6, 7, 8, and 9, plus Table 5
bash scripts/run-stage2-docker.sh --experiment=figures-6-7-8-9-table5

bash scripts/run-stage2-docker.sh --experiment=figure-10
bash scripts/run-stage2-docker.sh --experiment=figure-11
bash scripts/run-stage2-docker.sh --experiment=figure-13
```

The launcher prints `STAGE2_RUN_ID` and `STAGE2_RESULTS_DIR`. To resume an
interrupted experiment, keep its inputs/image/hardware settings unchanged:

```bash
RUN_ID=previous-run-id \
  bash scripts/run-stage2-docker.sh --experiment=figure-10 --resume
```

A successful execution ends with `STAGE2_STATUS=PASS`. Results and diagnostic
logs remain under `$STAGE2_STATE_ROOT/runs/<run-id>/`; successfully completed
workload images are removed. `--preflight-only` checks inputs, tools, SSH,
and FPGA device access without building guest images or programming the FPGA.
Table 4 timing and TVM/Verilator are not supported in this image; the plotting
step continues to use the published Table 4 reference.

The existing host scripts remain available for the legacy workflow. The
container uses the [validated orchestration snapshot](docker/stage2/workflow/README.md).

## 5. Generate the paper outputs

```bash
source ~/.bashrc
cd ~/pytorch-chipyard-ae

# Point at one experiment's results, or a directory merging distinct workloads
# from the completed experiment runs.
bash scripts/run-plot.sh --results-dir="$STAGE2_RESULTS_DIR"
```

Set `STAGE2_RESULTS_DIR` to the path printed by the launcher (the printed value
is not automatically exported into your shell). When combining experiments,
copy each workload directory into a common result directory, preserving its
`model.log` and `autotune.log`. For the simple experiment, set
`PYTORCH_CHIPYARD_SIMPLE_STAGE2=1` when running the plot command.

Generated figures use the semantic filenames referenced by the paper source
under `scripts/figures/`. The plotting workflow checks every generated plot
referenced by the paper, prints each path, and ends with `FIGURES_STATUS=PASS`.
The same command validates the published Table 4 reference values in
`reference-results/table4.csv` and writes the final automatic output to
`scripts/figures/table4.csv`. Table 4 measures host wall-clock turnaround, so
the published reference is used for the final output to avoid reporting
machine-load and CPU-contention differences as compiler or simulator changes.

## 6. Clean up

The command below cleans the checkout according to the legacy cleanup policy.
Stage 2 Docker results live outside the checkout, in the run directory printed
by the launcher, and are retained.

```bash
bash scripts/clean.sh --yes
``` 
