# PyTorch-Chipyard Artifact

A reusable compiler-stack foundation for ML research on Chipyard hardware.
This README provides the end-to-end commands for reproducing the reported
results on a FireSim-enabled local FPGA server.

## External prerequisites

- Chipyard 1.12.3
- FireSim 1.20.1
- FireMarshal 1.12.0
- Vivado 2021.1
- XRT 2.16.204
- Verilator 5.022
- RISC-V GNU Toolchain 12.2.0
- Gemmini based on the gemmini-microtvm branch (commit 3b07a14b).

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

## 1. Unpack and build the Stage 1 image

Unpack the Zenodo archive in the home directory so that the artifact root is
`~/pytorch-chipyard`, then enter that directory:

Because the Docker image build and Stage 2 simulations can take a long time, run them inside a `tmux` session so that they continue if the SSH connection is interrupted.

```bash
# or Download it from Zenodo
git clone --branch v1.0.0 --depth 1 \
  https://github.com/crigg8/pytorch-chipyard-ae.git \
  pytorch-chipyard
cd ~/pytorch-chipyard
bash scripts/download-bitstreams.sh

source ~/.bashrc

docker build \
  -f docker/stage1.Dockerfile \
  -t pytorch-chipyard:stage1 \
  .
```

The Docker image only builds the compiler environment used by Stage 1. Stage 2
uses the externally installed FPGA host environment.

## 2. Run the bounded smoke test

Before starting the full evaluation, compile the shared 32x32x32 GEMM for the three PyTorch-Chipyard targets inside the Stage 1 image:

```bash
mkdir -p results

docker run --rm -it \
  -v "$PWD/results:/opt/pytorch-chipyard/results" \
  -v pytorch-chipyard-triton-cache:/tmp/triton-chipyard-cache \
  pytorch-chipyard:stage1 \
  bash scripts/run-smoke-test-stage1.sh
```

The command compiles separate RVV, Gemmini, and scalar artifacts and prints
`SMOKE_STAGE1_STATUS=PASS`. Complete the smoke test on the FPGA host with:

```bash
source ~/.bashrc
cd ~/pytorch-chipyard

bash scripts/run-smoke-test.sh
```

The test checks the PyTorch-Chipyard Docker build, local FPGA setup, and TVM installation.
It ends with `SMOKE_TEST_STATUS=PASS`. 

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

## 4. Run Stage 2

Stage2 builds the FireMarshal images and runs the FireSim simulations. Although the entire workflow can be executed at once using ```run-stage2.sh```, it takes approximately 7--10 days; therefore, running it in experiment units is recommended. The simple test is a shorter 2--3 day workflow that produces a subset of the figures.

```bash
# Simple test: Partial Figures 6ab, 8a, 9, 11, 13ac
bash scripts/simple-stage2.sh

# Figures 7, 8, and 9, plus Table 5
bash scripts/run-stage2.sh --experiment=figures-7-8-9-table5

# Figure 10
bash scripts/run-stage2.sh --experiment=figure-10

# Figure 11
bash scripts/run-stage2.sh --experiment=figure-11

# Figure 13
bash scripts/run-stage2.sh --experiment=figure-13

# Table 4
bash scripts/run-stage2.sh --experiment=table-4
```

FireSim retains only `model.log` and `autotune.log` under
`~/pytorch-chipyard/scripts/figures/results-workload/<workload>/`. 
A successful run prints the durable paths and ends with
`STAGE2_STATUS=PASS`.

## 5. Generate the paper outputs

```bash
source ~/.bashrc
cd ~/pytorch-chipyard

bash scripts/run-plot.sh
```

Generated figures use the semantic filenames referenced by the paper source
under `scripts/figures/`. The plotting workflow checks every generated plot
referenced by the paper, prints each path, and ends with `FIGURES_STATUS=PASS`.
A complete Table 4 run writes `scripts/figures/table4.csv` and
`scripts/figures/table4_rows.tex`. 

## 6. Clean up

Below command removes everything in `pytorch-chipyard` except the final files under `scripts/figures/`.

```bash
bash scripts/clean.sh --yes
``` 
