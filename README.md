# Smith–Waterman Sequence Alignment using OpenMP

A parallel programming project implementing local biological sequence alignment using the Smith–Waterman algorithm in C.

## Implementations

- `sw_serial.c` — serial baseline implementation.
- `sw_parallel.c` — tiled wavefront parallel implementation using OpenMP.
- `sw_parallel_simd.c` — experimental SIMD implementation.
- `sw_parallel_tlp.c` — additional parallel implementation.
- `sw_parallel_wavefront.c` — wavefront implementation.

## Build

Requirements: GCC, Make, and OpenMP support.

```bash
make




The project uses optimization flags including -O3, -march=native, and -funroll-loops. OpenMP is enabled for parallel builds using -fopenmp.
Run
Check the source code for the accepted command-line arguments.
Example:
./sw_serial 8000 8000
./sw_parallel 8000 8000


Correctness and Performance
The evidence/, evidence_serial_matched/, and submission/ directories contain correctness results, benchmark timings, compiler optimization reports, and system information.
The benchmark results compare serial and parallel execution across thread counts. The SIMD implementation is experimental; refer to the recorded measurements rather than assuming it improves performance.
Project Report
See submission/filled_smith_waterman_report.md for the detailed Markdown report.
Team
- CB.SC.U4CSE23702 — Adhi Krishna
- CB.SC.U4CSE23759 — Balaji Ar.
- CB.SC.U4CSE23769 — Praneeth G
- CB.SC.U4CSE23204 — Adwaith A
  EOF

