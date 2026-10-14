# Stage 2 runtime patches

Apply these patches only to the commits in `../sources.lock`.

- `firesim.patch`: per-run deploy paths and prebuilt driver support from the
  hardware-tested AE environment. Hardware generation changes are excluded.
- `firemarshal.patch`: RVV guest configuration, deterministic guest address
  layout, and runtime Python/toolchain selection relative to the install root.
- `linux-rvv.patch`: the AE kernel's RVV register handling used by the successful
  U250 smoke run. Preserve this with the matching prebuilt RVV hardware.

`patch-runtime.py` and `patch-local.py` then add guestfish disk access,
container-local simulation, and host FPGA programming over SSH. Vivado and the
XDMA kernel module are not installed into the image.
