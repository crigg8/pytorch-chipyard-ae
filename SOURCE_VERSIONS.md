# Source Versions

## Docker build environment

| Component | Version |
| --- | --- |
| Ubuntu | 20.04 |
| Miniconda | Python 3.12, Miniconda 24.11.1 |
| Conda | 24.11.3 |
| Python | 3.12 |
| GCC/G++ | 13 |
| conda-lock | 1.4.0 |
| PyTorch | 2.8.0 |
| torchvision | 0.23.0 |
| transformers | 4.56.2 |
| sentencepiece | 0.2.0 |

## TVM-Gemmini baseline

| Component | Revision or version |
| --- | --- |
| TVM, `gemmini-microtvm` branch | `3b07a14bb936be64115403d839f5ef002ca8102d` |
| Gemmini | `b6bdad59cbd6313f1ea4c93d3493db3d59b9e418` |
| gemmini-rocc-tests | `ae0cd8236d32fccf7197a7ac0634df5513cec4db` |
| ESP/Spike ISA simulator | `051d820f08be84d069993de4375d29c91eb2f577` |
| Python | 3.10 |
| LLVM used by TVM | 12 |
| NumPy | 1.23.5 |
| TensorFlow CPU | 2.12.1 |
| TensorFlow Estimator | 2.12.0 |
| TFLite | 2.10.0 |

## External FPGA host environment

| Component | Version |
| --- | --- |
| Chipyard | 1.12.3 |
| FireSim | 1.20.1 |
| FireMarshal | 1.12.0 |
| Vivado | 2021.1 |
| XRT | 2.16.204 |
| Verilator | 5.022 |
| RISC-V GNU Toolchain | 12.2.0 |


## Stage 2 Docker integration

- AE compiler baseline: commit
  `cd48c3e2cdfeb960588aa081316f68264e761769`.
- Host programming: Vivado **2021.1** and its `hw_server`, with host XDMA.
- Container runtime: Ubuntu 20.04, FireSim 1.20.1, FireMarshal 1.12.0,
  Python 3.10.14, and RISC-V guest-build GCC 13.2.0. Source commits are
  pinned in [sources.lock](docker/stage2/sources.lock); packages are pinned in
  [conda-linux-64.lock](docker/stage2/conda-linux-64.lock) and
  [requirements.txt](docker/stage2/requirements.txt).
- Guest input ELF linking: RISC-V Linux GCC **12.2.0**, bundled in Stage 1
  from `ucb-bar/riscv-tools 1.0.3`. Package URLs and hashes are pinned in
  [riscv-linux-64.lock](docker/stage1/riscv-linux-64.lock).
- Hardware-tested image:
  `sha256:aa40c7329844a5208ed32440b33f1f98b4dbfb547b59055eb794b4b4676f1433`.
  Its three-workload smoke evidence is in
  [validation.json](docker/stage2/validation.json); this does not certify
  subsequent image rebuilds or full-model experiments.
- The five container orchestration scripts are preserved under
  [docker/stage2/workflow](docker/stage2/workflow/README.md), with SHA-256
  verification during the image build.
- Runtime patches are checked in under `docker/stage2/patches/`. The image
  records source/support and FPGA archive hashes in `/opt/stage2-metadata/`.
  No installed host Chipyard or Conda environment is copied.

