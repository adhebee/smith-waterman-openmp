# Capstone Report: Exploring ILP, DLP, and TLP in Smith–Waterman Sequence Alignment using OpenMP

**Student name:** [Enter your name]  
**Domain(s) covered:** Bioinformatics / sequence alignment; Instruction-Level Parallelism (ILP), Data-Level Parallelism (DLP), and Thread-Level Parallelism (TLP)  
**Date:** 13 October 2026

---

## 1. Abstract

Smith–Waterman is a dynamic-programming algorithm for finding the highest-scoring local alignment between two biological sequences. It is used in bioinformatics applications such as comparing DNA, RNA, or protein sequences, identifying similar regions, and supporting sequence-analysis pipelines. As sequence databases grow, alignment throughput becomes important because the dynamic-programming matrix requires quadratic work with respect to sequence length. This project implemented a serial baseline and an OpenMP tiled-wavefront version. The parallel version processes tiles along dependency-respecting wavefronts, allowing independent tiles on the same wavefront to execute concurrently. On sequences of length 8,000, all tested thread counts returned the same maximum score, 3579, as the serial implementation. In the repeated OpenMP benchmark, median elapsed time decreased from 0.27 seconds with one thread to 0.13 seconds with four and eight threads, giving approximately 2.08× speedup relative to the one-thread OpenMP version. A separate SIMD experiment used an OpenMP SIMD reduction over cells on local anti-diagonals. GCC generated vector instructions for that loop, but its median runtime was 0.21 seconds at four threads, slower than the tiled baseline's 0.16 seconds. The experiment demonstrates that thread-level parallelism and vectorization are distinct optimizations, and that compiler-reported vectorization does not guarantee faster end-to-end execution.

---

## 2. Introduction & Domain Motivation

### Real-world application

Smith–Waterman performs local sequence alignment: it finds the best matching region between two sequences rather than requiring the entire sequences to align. Concrete applications include comparing DNA sequences in genomics, comparing protein sequences to help identify related proteins, and searching biological databases for local similarities. These comparisons can support tasks such as identifying conserved regions or investigating whether an unknown sequence resembles a known sequence.

### Why performance matters

The dynamic-programming matrix contains one cell for each pair of sequence positions, so the work grows approximately as \(O(nm)\) for sequence lengths \(n\) and \(m\). Large batches of sequence comparisons can therefore consume substantial computation time. Faster alignment can improve throughput in bioinformatics pipelines and reduce the compute time needed for large-scale analyses. This project focuses on throughput for one pair of long sequences, rather than a hard real-time latency requirement.

### Problem and correctness criteria

The program aligns two generated sequences using match score +2, mismatch score −1, and gap penalty −2. The local-alignment recurrence is:

\[
H(i,j)=\max\left(0,\ H(i-1,j-1)+s(i,j),\ H(i-1,j)-2,\ H(i,j-1)-2\right)
\]

where \(s(i,j)=2\) for a match and −1 for a mismatch. The result reported is the maximum score in the matrix. The principal benchmark used two sequences of length 8,000 (an 8,000 × 8,000 dynamic-programming matrix). The serial and parallel versions both reported `max_score=3579` at this size. They also agreed on smaller tests, including length 1,000 (score 444) and length 4,000 (score 1797). These checks support correctness for the tested inputs; they do not prove correctness for every possible input.

---

## 3. Background: ILP, DLP, TLP

**Instruction-Level Parallelism (ILP).** ILP means that a processor can overlap or execute independent machine instructions at the same time, for example through pipelining or out-of-order execution. In this project, ILP is mainly exposed by the compiler and CPU rather than by a dedicated OpenMP directive. Compiler optimization, loop unrolling, and the target architecture can help expose independent instructions, although dependencies in the dynamic-programming recurrence limit how much of the computation can be reordered.

**Data-Level Parallelism (DLP).** DLP means applying the same operation to multiple data elements in parallel, commonly using SIMD vector instructions. The experimental implementation adds `#pragma omp simd reduction(max:tile_max)` to an inner loop over cells on a local anti-diagonal. GCC's optimization report confirms that this loop was vectorized using 32-byte and 16-byte vectors. This does not mean that every loop in the algorithm was vectorized.

**Thread-Level Parallelism (TLP).** TLP means distributing independent work across multiple CPU threads. The main OpenMP implementation divides the matrix into tiles and processes tile wavefronts in dependency order. Tiles on the same wavefront can be processed concurrently, while the order between wavefronts preserves the data dependencies of the recurrence.

| Concept | OpenMP construct / mechanism | Where in the code |
|---|---|---|
| TLP | OpenMP parallel region and work-sharing across tiles on a wavefront | `sw_parallel.c`, tiled wavefront loop |
| DLP | `#pragma omp simd reduction(max:tile_max)` in the experimental implementation | `sw_parallel_simd.c`, inner local anti-diagonal loop |
| ILP | No dedicated OpenMP construct; compiler/CPU driven | GCC flags `-O3 -march=native -funroll-loops`; inspect vectorization reports and generated code |

---

## 4. Baseline (Unoptimized) Implementation

The serial implementation stores the dynamic-programming matrix in a flat array and fills it in row-major order. Each cell depends on the cell above, the cell to the left, and the upper-left diagonal cell. It tracks the maximum score encountered.

Simplified pseudocode:

```text
maximum = 0
for i = 1 ... n:
    for j = 1 ... m:
        score = max(0,
                    H[i-1,j-1] + substitution_score,
                    H[i-1,j] - gap_penalty,
                    H[i,j-1] - gap_penalty)
        H[i,j] = score
        maximum = max(maximum, score)
```

The serial program's internal timer reported 0.3854 seconds for an 8,000 × 8,000 case in the correctness run. A separate five-run timing batch on the native Linux project directory had external elapsed times of 0.64, 0.49, 0.52, 0.41, and 0.43 seconds, giving a median of 0.49 seconds. The internal timer and external `/usr/bin/time` measurements use different timing methods and should not be mixed as though they were the same measurement. The parallel scaling table below uses the repeated external elapsed-time results for the OpenMP versions and labels the serial comparison as a separate batch.

---

## 5. Optimization Strategy

### Stage 1 — Serial baseline

1. **Change:** Implemented the recurrence using a flat dynamic-programming matrix and row-major traversal.
2. **Target:** Establishes the correctness and performance baseline; it does not explicitly use TLP or DLP.
3. **Correctness/thread safety:** The serial program is single-threaded. Both sequences are generated from fixed seeds for repeatable test cases.

### Stage 2 — Tiled OpenMP wavefront (TLP)

1. **Change:** Divided the matrix into tiles and processed tiles by wavefront/diagonal. Tiles on the same wavefront are eligible to run concurrently, while dependent wavefronts execute in order.
2. **Target:** TLP, using OpenMP threads to process independent tiles.
3. **Correctness/thread safety:** The wavefront order preserves inter-tile dependencies. Cells within a tile are computed in dependency-respecting order. The program's generated input and score result were checked against the serial implementation. All tested thread counts returned score 3579 for the 8,000 × 8,000 input.

### Stage 3 — TLP plus experimental DLP (SIMD)

1. **Change:** Created `sw_parallel_simd.c` as a separate experiment. Within each tile, cells are traversed on local anti-diagonals, and an OpenMP SIMD reduction is applied to the independent loop over a diagonal.
2. **Target:** DLP through `omp simd`, while retaining TLP across tile wavefronts.
3. **Correctness/thread safety:** The SIMD experiment returned score 3579 for the 8,000 × 8,000 input and score 444 for the 1,000 × 1,000 input, matching the serial output in those tests. The baseline source `sw_parallel.c` was kept unchanged. The experiment was slower in the measured four-thread comparison, so it is not selected as the main implementation.

### Stage 4 — Compiler-driven ILP

1. **Change:** Compiled with `-O3 -march=native -funroll-loops` and generated GCC vectorization diagnostics.
2. **Target:** Compiler and hardware optimization can expose ILP and other low-level optimizations; no claim is made that these flags guarantee a particular ILP speedup.
3. **Correctness/thread safety:** The optimized serial and parallel programs were checked for matching maximum scores. GCC reports indicate that the dependency-heavy baseline DP loops were not vectorized because of possible data dependencies. The SIMD experiment's explicitly annotated anti-diagonal loop was vectorized, but this did not improve overall runtime.

---

## 6. Experimental Setup

The following information was recorded during the project. Items marked **verify/add** should be confirmed from the machine used for the final benchmark before submission.

- **CPU:** Intel Core i5-11400H @ 2.70 GHz
- **Physical cores / logical CPUs:** 6 physical cores / 12 logical CPUs
- **Cache:** L2 7.5 MiB and L3 12 MiB were reported. **Verify L1 cache sizes** using `lscpu -C` or `lscpu`.
- **RAM:** Approximately 3.7 GiB total was visible to the WSL environment in the earlier system check. **Re-run `free -h` and report the current output** if this environment has changed.
- **OS / kernel:** Ubuntu 22.04.5 LTS under WSL2; kernel `6.18.54.1-microsoft-standard-WSL2`
- **Compiler:** GCC 11.4.0
- **OpenMP:** `_OPENMP 201511`, corresponding to OpenMP 4.5
- **Main serial compile flags:** `-O3 -march=native -Wall -funroll-loops`
- **Main parallel compile flags:** `-O3 -march=native -Wall -funroll-loops -fopenmp`
- **SIMD experiment compile flags:** `-O3 -march=native -Wall -Wextra -fopenmp -fopt-info-vec-all`
- **Input sizes:** 1,000 × 1,000; 4,000 × 4,000; and 8,000 × 8,000. The principal performance comparison used 8,000 × 8,000.
- **Benchmark method:** Five repetitions per OpenMP thread count, using `/usr/bin/time -f 'elapsed=%e s'`; thread counts were 1, 2, 4, and 8. Medians are reported to reduce the effect of individual noisy runs.

The main source was built with:

```bash
make clean && make
```

The SIMD experiment was built separately with:

```bash
gcc -O3 -march=native -Wall -Wextra -fopenmp \
    -fopt-info-vec-all -o sw_parallel_simd sw_parallel_simd.c \
    2> evidence/vec_simd_report.txt
```

---

## 7. Results

### 7.1 OpenMP thread-count benchmark

For the serial version, the external elapsed-time median from the earlier five-run batch was 0.49 seconds. The OpenMP timings below are medians from the later five-run batch. Since these were collected in separate batches, speedups against serial are approximate and should be interpreted with that limitation in mind.

| Version | Threads | Median time (s) | Speedup (vs. serial median 0.49 s) | Parallel Efficiency (%) | Notes |
|---|---:|---:|---:|---:|---|
| Serial | 1 | 0.49 | 1.00× | — | Five-run external median; separate batch |
| +TLP | 1 | 0.27 | 1.81× | 181.5%* | One-thread OpenMP reference; not a true serial implementation |
| +TLP | 2 | 0.17 | 2.88× | 144.1%* | All runs returned score 3579 |
| +TLP | 4 | 0.13 | 3.77× | 94.2%* | Best observed median runtime |
| +TLP | 8 | 0.13 | 3.77× | 47.1%* | No clear improvement over four threads |
| +TLP + DLP | 4 | 0.21 | 2.33× | 58.3%* | SIMD experiment; slower than tiled baseline |

**Important interpretation note:** The template defines parallel efficiency as speedup divided by thread count. The values marked with `*` above use the separate serial median of 0.49 seconds and therefore are not a valid conventional parallel-efficiency comparison: values over 100% indicate that the serial and parallel batches are not directly comparable, and the one-thread OpenMP version has a different implementation overhead. For a consistent scaling calculation, use the one-thread OpenMP median of 0.27 seconds as the reference:

| Threads | Median time (s) | Speedup vs. 1-thread OpenMP | Parallel efficiency |
|---:|---:|---:|---:|
| 1 | 0.27 | 1.00× | 100% |
| 2 | 0.17 | 1.59× | 79.4% |
| 4 | 0.13 | 2.08× | 51.9% |
| 8 | 0.13 | 2.08× | 26.0% |

All serial, tiled OpenMP, and SIMD test outputs matched on the tested input sizes. For the main 8,000 × 8,000 case, the maximum score was 3579.

### 7.2 Vectorization evidence

The standard tiled implementation's vectorization reports showed that the dependency-heavy dynamic-programming loops were not vectorized; GCC reported possible dependencies between data references. The SIMD experiment explicitly annotated an inner anti-diagonal loop. GCC reported:

```text
sw_parallel_simd.c:112:29: optimized: loop vectorized using 32 byte vectors
sw_parallel_simd.c:112:29: optimized: loop vectorized using 16 byte vectors
```

This demonstrates vectorization of the annotated loop, not the entire algorithm. The SIMD experiment's median elapsed time was 0.21 seconds, compared with 0.16 seconds for the four-thread tiled baseline in the separate five-run comparison. Therefore, the vectorization did not improve total performance in this experiment.

### 7.3 `perf stat` evidence

A supplementary `perf stat` comparison was collected earlier on the Windows-mounted project directory, not during the native Linux benchmark batch. These values must be labeled separately because the environment and timing conditions differed.

| Version | Instructions | Cycles | Instructions/cycle (IPC) |
|---|---:|---:|---:|
| Serial | 1,253,066,027 | 755,808,788 | 1.66 |
| OpenMP, 8 threads | 1,334,253,135 | 3,027,314,228 | 0.44 |

IPC is calculated as instructions divided by cycles. The parallel process's aggregate cycle count reflects work across threads and cannot be directly compared with serial cycles as a single-core efficiency measure. Re-run `perf stat` in the final native directory if a directly comparable profiling result is required.

---

## 8. Discussion

### Where did scaling fall short of ideal?

The one-thread OpenMP median was 0.27 seconds, while the four-thread median was 0.13 seconds, giving approximately 2.08× speedup rather than ideal 4× speedup. The eight-thread median remained 0.13 seconds, so the extra threads did not provide a measurable gain. Smith–Waterman has dependencies between neighboring cells; the tiled wavefront permits parallel execution only among tiles that are ready on the same wavefront. Some wavefronts expose less parallel work than others, which can lead to load imbalance. Synchronization and scheduling also have a cost. Memory traffic and cache behavior may contribute, but the available measurements do not establish memory bandwidth saturation or false sharing as the main cause. Amdahl's law also predicts that serial portions and coordination overhead limit total speedup, although this experiment does not isolate the serial fraction.

### Did DLP help as much as TLP?

No. The tiled OpenMP version had a four-thread median of 0.16 seconds in the direct comparison, while the SIMD experiment had a median of 0.21 seconds. The compiler did vectorize the annotated loop, but the algorithm's recurrence limits which cells can be evaluated independently. The experiment processes local anti-diagonals, many of which are relatively short, and incurs extra loop/reduction overhead. Those costs may outweigh the benefit of SIMD instructions. The result illustrates that vectorization evidence alone does not guarantee an end-to-end speedup.

### What limited or enabled ILP?

The build used `-O3 -march=native -funroll-loops`, allowing GCC to optimize for the host architecture and consider loop unrolling. However, the baseline DP loops were reported as not vectorized because GCC detected possible dependencies between data references. The experimental SIMD loop was vectorized using 32-byte and 16-byte vectors. ILP can still be exploited by the processor through pipelining and out-of-order execution, but no isolated hardware experiment measured ILP independently. Therefore, the report should describe compiler and CPU support for ILP without claiming a quantified ILP speedup.

### What would change at larger scale?

For larger sequences, I would benchmark multiple tile sizes and wavefront scheduling strategies, and profile cache misses and memory bandwidth to determine whether the algorithm becomes memory-bound. A GPU implementation could exploit parallelism across cells or tiles, but the recurrence dependencies would still need to be respected. For batches of independent sequence pairs, distributing separate alignments across CPU cores, GPUs, or MPI processes may expose more parallelism than a single alignment alone. Any larger-scale implementation should preserve score equivalence with the serial baseline and be evaluated with repeated runs and clearly recorded hardware/compiler settings.

---

## 9. Conclusion

The experiments show that TLP provided the clearest performance improvement: the tiled OpenMP implementation reduced median time from 0.27 seconds at one thread to 0.13 seconds at four threads. Increasing to eight threads did not improve the measured median, demonstrating diminishing returns. DLP was successfully exposed in a separate SIMD experiment, but that implementation was slower overall, showing that vectorization is not automatically beneficial when loop structure and data dependencies add overhead. ILP was mainly left to GCC and the CPU; the compiler reports show that dependencies limited vectorization in the baseline loops. Together, the results demonstrate that TLP, DLP, and ILP are distinct optimization mechanisms that must be assessed with both correctness checks and measured performance.

---

## 10. References

1. Smith, T. F., and Waterman, M. S. “Identification of Common Molecular Subsequences.” *Journal of Molecular Biology*, 147(1), 1981, pp. 195–197. https://doi.org/10.1016/0022-2836(81)90087-5
2. OpenMP Architecture Review Board. *OpenMP API Specification*, OpenMP 4.5. https://www.openmp.org/specifications/
3. GNU Project. *GCC Manual: Optimize Options and Developer Options (optimization/vectorization reports).* https://gcc.gnu.org/onlinedocs/
4. Linux man-pages project. `perf-stat(1)` manual page. https://man7.org/linux/man-pages/man1/perf-stat.1.html

---

## Appendix

Attach or paste the final source code for `sw_serial.c` and `sw_parallel.c` here, including comments explaining the recurrence, tile dependencies, and OpenMP synchronization. Include `sw_parallel_simd.c` as the experimental DLP version if permitted by the submission instructions. The source files were not embedded in this generated report because they were not available alongside the uploaded template in this session.

### Evidence files to include with submission

- `evidence/system_info.txt` — CPU, OS, kernel, and compiler details; add cache and RAM details before submission.
- `evidence/vec_report_all.txt` and `evidence/vec_serial_report.txt` — baseline vectorization diagnostics.
- `evidence/vec_simd_report.txt` — SIMD experiment vectorization diagnostics.
- `evidence/perf_serial.txt` and `evidence/perf_parallel.txt` — supplementary `perf stat` output from the earlier Windows-mounted project run.
- Terminal output showing matching scores and the five-run timing results for 1, 2, 4, and 8 threads.
