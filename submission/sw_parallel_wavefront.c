/*
 * Domain B: Smith-Waterman Sequence Alignment
 * OpenMP wavefront parallel implementation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MATCH     2
#define MISMATCH -1
#define GAP      -2

static int max4(int a, int b, int c, int d)
{
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    if (d > m) m = d;
    return m;
}

/* Same sequence generator as the supplied serial implementation. */
static void gen_sequence(char *seq, int len, unsigned int seed)
{
    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    srand(seed);

    for (int i = 0; i < len; i++)
        seq[i] = alphabet[rand() % 4];
}

int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 4000;
    int m = (argc > 2) ? atoi(argv[2]) : 4000;

    if (n <= 0 || m <= 0) {
        fprintf(stderr, "Sequence lengths must be positive.\n");
        return 1;
    }

    size_t rows = (size_t)n + 1;
    size_t cols = (size_t)m + 1;

    if (rows > (size_t)-1 / cols ||
        rows * cols > (size_t)-1 / sizeof(int)) {
        fprintf(stderr, "Matrix size is too large.\n");
        return 1;
    }

    char *seq1 = malloc((size_t)n);
    char *seq2 = malloc((size_t)m);
    int *H = calloc(rows * cols, sizeof(int));

    if (seq1 == NULL || seq2 == NULL || H == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(seq1);
        free(seq2);
        free(H);
        return 1;
    }

    gen_sequence(seq1, n, 42);
    gen_sequence(seq2, m, 1337);

    int maxScore = 0;
    double start = omp_get_wtime();

    /*
     * Wavefront traversal: cells on one anti-diagonal are independent.
     * All dependencies are on earlier anti-diagonals.
     */
    for (int d = 2; d <= n + m; d++) {
        int i_start = (d - 1 > m) ? d - 1 - m : 1;
        int i_end = (d - 1 < n) ? d - 1 : n;
        int diagonal_max = 0;

        #pragma omp parallel for reduction(max:diagonal_max) schedule(static)
        for (int i = i_start; i <= i_end; i++) {
            int j = d - i;
            size_t idx = (size_t)i * cols + (size_t)j;

            int score = (seq1[i - 1] == seq2[j - 1])
                        ? MATCH : MISMATCH;

            H[idx] = max4(
                0,
                H[(size_t)(i - 1) * cols + (size_t)(j - 1)] + score,
                H[(size_t)(i - 1) * cols + (size_t)j] + GAP,
                H[(size_t)i * cols + (size_t)(j - 1)] + GAP
            );

            if (H[idx] > diagonal_max)
                diagonal_max = H[idx];
        }

        if (diagonal_max > maxScore)
            maxScore = diagonal_max;
    }

    double elapsed = omp_get_wtime() - start;

    printf("[sw_parallel] n=%d m=%d time=%.4f s max_score=%d threads=%d\n",
           n, m, elapsed, maxScore, omp_get_max_threads());

    free(H);
    free(seq1);
    free(seq2);

    return 0;
}
