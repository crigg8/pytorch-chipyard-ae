# Validated container workflow

These five scripts preserve the orchestration used by Stage 2 image
`sha256:aa40c7329844a5208ed32440b33f1f98b4dbfb547b59055eb794b4b4676f1433`.
The RVV 4-core, Gemmini 4-core, and scalar 16-core smoke workloads completed
successfully on an Alveo U250 with host Vivado 2021.1 on 2026-10-14.

The interpreter error message now refers to the container runtime.

The image build verifies `SHA256SUMS` and installs these files into
`/opt/pytorch-chipyard/scripts/`.
The `v1.0.0` host scripts remain in the repository for the legacy host workflow.
Their cleanup behavior removes ELF derivatives and most diagnostic output;
the container workflow instead mounts input ELFs read-only and keeps logs and
checkpoints under the selected run directory.

Edit this directory when changing container orchestration, update the checksum
manifest, and rerun the Stage 2 tests. The compiler sources remain based on
the AE `v1.0.0` commit `cd48c3e2cdfeb960588aa081316f68264e761769`.
