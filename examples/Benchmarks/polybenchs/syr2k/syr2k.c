#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef N
# define N 1024
#endif
#ifndef M
# define M 1024
#endif

/* Default data type is double (dsyr2k). */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

DATA_TYPE alpha;
DATA_TYPE beta;
DATA_TYPE A[N][M];
DATA_TYPE B[N][M];
DATA_TYPE C[N][N];

static inline
void init_array() {
  int i, j;

  alpha = 12435;
  beta = 4546;
  for (i = 0; i < N; ) {
    for (j = 0; j < N; ) {
      C[i][j] = ((DATA_TYPE)i * j + 2) / N;
      j++;
    }
    for (j = 0; j < M; ) {
      A[i][j] = ((DATA_TYPE)i * j) / N;
      B[i][j] = ((DATA_TYPE)i * j + 1) / N;
      j++;
    }
    i++;
  }
}

/* Define the live-out variables. Code is not executed unless
 POLYBENCH_DUMP_ARRAYS is defined. */
static inline
void print_array(int argc, char** argv) {
  int i, j;
#ifndef POLYBENCH_DUMP_ARRAYS
  if(argc > 42 && !strcmp(argv[0], ""))
#endif
  {
    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++) {
        fprintf(stderr, DATA_PRINTF_MODIFIER, C[i][j]);
        if((i * N + j) % 80 == 20)
          fprintf(stderr, "\n");
      }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int i, j, k;
  int n = N;
  int m = M;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  timer_start();

  /*    C := alpha*A*B' + alpha*B*A' + beta*C */
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] *= beta;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < m; k++) {
        C[i][j] += alpha * A[i][k] * B[j][k];
        C[i][j] += alpha * B[i][k] * A[j][k];
      }

  /* Stop and print timer. */
  timer_stop_display(); ;

  print_array(argc, argv);

  return 0;
}
