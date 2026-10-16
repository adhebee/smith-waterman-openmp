
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define MATCH 2
#define MISMATCH (-1)
#define GAP (-2)
#define TILE_SIZE 64

static inline int max4(int a, int b, int c, int d)
{
    int x = (a > b) ? a : b;
    int y = (c > d) ? c : d;
    int z = (x > y) ? x : y;
    return (z > 0) ? z : 0;
}

static void generate_sequence(char *seq, int len, unsigned int seed)
{
    const char alphabet[] = {'A', 'C', 'G', 'T'};

    srand(seed);
    for (int i = 0; i < len; i++)
        seq[i] = alphabet[rand() % 4];
}

int main(int argc, char **argv)
{
    int n = 4000;
    int m = 4000;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) m = atoi(argv[2]);

    if (n <= 0 || m <= 0) {
        fprintf(stderr, "Sequence lengths must be positive.\n");
        return 1;
    }

    size_t rows = (size_t)n + 1;
    size_t cols = (size_t)m + 1;

    if (rows > SIZE_MAX / cols ||
        rows * cols > SIZE_MAX / sizeof(int)) {
        fprintf(stderr, "Matrix size is too large.\n");
        return 1;
    }

    char *seq1 = malloc((size_t)n);
    char *seq2 = malloc((size_t)m);
    int *H = calloc(rows * cols, sizeof(int));

    if (!seq1 || !seq2 || !H) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(seq1);
        free(seq2);
        free(H);
        return 1;
    }

    generate_sequence(seq1, n, 42);
    generate_sequence(seq2, m, 1337);

    int tile_rows = (n + TILE_SIZE - 1) / TILE_SIZE;
    int tile_cols = (m + TILE_SIZE - 1) / TILE_SIZE;
    int maxScore = 0;
    int diag_max = 0;

    double start = omp_get_wtime();

    #pragma omp parallel shared(H, seq1, seq2, maxScore, diag_max)
    {
        for (int td = 0; td < tile_rows + tile_cols - 1; td++) {

            #pragma omp single
            {
                diag_max = 0;
            }

            int bi_start = td - (tile_cols - 1);
            if (bi_start < 0) bi_start = 0;

            int bi_end = td;
            if (bi_end >= tile_rows) bi_end = tile_rows - 1;

            #pragma omp for reduction(max:diag_max) schedule(static)
            for (int bi = bi_start; bi <= bi_end; bi++) {
                int bj = td - bi;

                int i_start = bi * TILE_SIZE + 1;
                int i_end = (bi + 1) * TILE_SIZE;
                if (i_end > n) i_end = n;

                int j_start = bj * TILE_SIZE + 1;
                int j_end = (bj + 1) * TILE_SIZE;
                if (j_end > m) j_end = m;

                int tile_max = 0;

                int tile_height = i_end - i_start + 1;
                int tile_width = j_end - j_start + 1;

                for (int wave = 0; wave < tile_height + tile_width - 1; wave++) {
                    int local_i_start = wave - (tile_width - 1);
                    if (local_i_start < 0) local_i_start = 0;

                    int local_i_end = wave;
                    if (local_i_end >= tile_height) local_i_end = tile_height - 1;

                    #pragma omp simd reduction(max:tile_max)
                    for (int local_i = local_i_start;
                         local_i <= local_i_end; local_i++) {
                        int local_j = wave - local_i;
                        int i = i_start + local_i;
                        int j = j_start + local_j;
                        size_t row = (size_t)i * cols;

                        int score_diag =
                            H[row - cols + (size_t)j - 1] +
                            (seq1[i - 1] == seq2[j - 1]
                                ? MATCH : MISMATCH);
                        int score_up = H[row - cols + (size_t)j] + GAP;
                        int score_left = H[row + (size_t)j - 1] + GAP;
                        int score = max4(score_diag, score_up, score_left, 0);

                        H[row + (size_t)j] = score;
                        if (score > tile_max) tile_max = score;
                    }
                }

                if (tile_max > diag_max)
                    diag_max = tile_max;
            }

            #pragma omp single
            {
                if (diag_max > maxScore)
                    maxScore = diag_max;
            }
        }
    }

    double elapsed = omp_get_wtime() - start;

    printf("[sw_parallel] n=%d m=%d time=%.6f max_score=%d threads=%d\n",
           n, m, elapsed, maxScore, omp_get_max_threads());

    free(seq1);
    free(seq2);
    free(H);

    return 0;
}
