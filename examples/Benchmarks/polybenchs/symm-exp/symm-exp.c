#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef N
# define N 512
#endif
#ifndef M
# define M 512
#endif

/* Default data type is double (dsymm). */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
DATA_TYPE alpha;
DATA_TYPE beta;
DATA_TYPE acc[M][N];
DATA_TYPE A[N][N];
DATA_TYPE B[M][N];
DATA_TYPE C[M][N];

static void init_array() {
  int i, j;

  alpha = 12435;
  beta = 4546;
  for (i = 0; i < N;) {
    for (j = 0; j < N;) {
      A[i][j] = ((DATA_TYPE)i * j) / N;
      j++;
    }
    i++;
  }
  for (i = 0; i < M;) {
    for (j = 0; j < N;) {
      B[i][j] = ((DATA_TYPE)i * j + 1) / N;
      C[i][j] = ((DATA_TYPE)i * j + 2) / N;
      j++;
      i++;
    }
  }
}

/* Define the live-out variables. Code is not executed unless
 POLYBENCH_DUMP_ARRAYS is defined. */
static void print_array(int argc, char** argv) {
  int i, j;
#ifndef POLYBENCH_DUMP_ARRAYS
  if(argc > 42 && !strcmp(argv[0], ""))
#endif
  {
    for (i = 0; i < M; i++)
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

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    init_array();
  }

#ifdef PGI_ACC
#pragma acc region
{
#endif
#ifdef PGCC
#pragma scop
#endif
  /*  C := alpha*A*B + beta*C, A is symetric */
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      acc[i][j] = 0;
      for (k = 0; k < j - 1; k++) {
        C[k][j] += alpha * A[k][i] * B[i][j];
        acc[i][j] += B[k][j] * A[k][i];
      }
      C[i][j] = beta * C[i][j] + alpha * A[i][i] * B[i][j] + alpha * acc[i][j];
    }
#ifdef PGCC
#pragma endscop
#endif
#ifdef PGI_ACC
}
#endif

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display(); ;

  print_array(argc, argv);

  return 0;
}
