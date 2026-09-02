# Reference results

`table4.csv` contains the values reported in Table 4 of the paper. These are
published reference values, not measurements collected during the current
artifact-evaluation run. Table 4 reports host wall-clock turnaround, so its
Verilator values are sensitive to CPU load and host contention.

`scripts/run-plot.sh` validates this file and materializes it as the final
automatic output `scripts/figures/table4.csv`.
