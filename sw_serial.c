/*
 * Domain B: Local Sequence Alignment - Smith-Waterman (SERIAL / UNOPTIMIZED baseline)
 *
 * Classic row-major dynamic programming. Each cell H[i][j] depends on
 * H[i-1][j-1], H[i-1][j], H[i][j-1] - a strict dependency chain that
 * prevents naive row/column parallelization (see sw_parallel.c for the fix).
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATCH     2
#define MISMATCH -1
#define GAP      -2

static int max4(int a, int b, int c, int d) {
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    if (d > m) m = d;
    return m;
}

int smith_waterman_serial(const char *seq1, const char *seq2, int n, int m) {
    int **H = malloc((n+1) * sizeof(int*));
    for (int i = 0; i <= n; i++) H[i] = calloc(m+1, sizeof(int));

    int maxScore = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            int s = (seq1[i-1] == seq2[j-1]) ? MATCH : MISMATCH;
            H[i][j] = max4(0, H[i-1][j-1]+s, H[i-1][j]+GAP, H[i][j-1]+GAP);
            if (H[i][j] > maxScore) maxScore = H[i][j];
        }
    }
    for (int i = 0; i <= n; i++) free(H[i]);
    free(H);
    return maxScore;
}

/* Generates a pseudo-random DNA sequence of given length (fixed seed => reproducible) */
static void gen_sequence(char *seq, int len, unsigned int seed) {
    const char alphabet[4] = {'A','C','G','T'};
    srand(seed);
    for (int i = 0; i < len; i++) seq[i] = alphabet[rand() % 4];
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 4000;
    int m = (argc > 2) ? atoi(argv[2]) : 4000;

    char *seq1 = malloc(n);
    char *seq2 = malloc(m);
    gen_sequence(seq1, n, 42);
    gen_sequence(seq2, m, 1337);

    clock_t start = clock();
    int score = smith_waterman_serial(seq1, seq2, n, m);
    clock_t end = clock();

    printf("[sw_serial] n=%d m=%d time=%.4f s max_score=%d\n",
           n, m, (double)(end-start)/CLOCKS_PER_SEC, score);

    free(seq1); free(seq2);
    return 0;
}
