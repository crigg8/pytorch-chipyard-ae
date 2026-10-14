#!/usr/bin/env bash
set -euo pipefail
source /opt/riscv-gcc-12.2.0/env.sh
[[ "$(riscv64-unknown-linux-gnu-g++ -dumpfullversion)" == 12.2.0 ]]
temporary="$(mktemp -d)"
trap 'rm -rf -- "$temporary"' EXIT
cat > "$temporary/check.cpp" <<'EOF'
#include <omp.h>
#include <iostream>
int main() {
  int count = 0;
  #pragma omp parallel reduction(+:count)
  count += 1;
  #ifdef CHECK_RVV
  asm volatile("vsetvli zero, zero, e32, m1, ta, ma");
  #endif
  std::cout << count << std::endl;
  return omp_get_max_threads() < 1;
}
EOF
for backend in scalar gemmini rvv; do
  flags=(-march=rv64imafdc -mabi=lp64d)
  if [[ "$backend" == rvv ]]; then
    flags=(-march=rv64imafdcv_zicsr_zifencei_zvl128b -mabi=lp64d -DCHECK_RVV)
  fi
  riscv64-unknown-linux-gnu-g++ "${flags[@]}" -fopenmp "$temporary/check.cpp" -o "$temporary/$backend.elf"
  readelf -h "$temporary/$backend.elf" | grep -q 'RISC-V'
  readelf -d "$temporary/$backend.elf" | grep -q 'libgomp.so.1'
  readelf -d "$temporary/$backend.elf" | grep -q 'libstdc++.so.6'
done
printf 'RISCV_GCC_12_2_LINK_STATUS=PASS\n'
