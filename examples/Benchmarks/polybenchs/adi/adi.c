#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef TSTEPS
# define TSTEPS 10
#endif
#ifndef N
# define N 1024
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif


DATA_TYPE X[N][N];
DATA_TYPE A[N][N];
DATA_TYPE B[N][N];

static void init_array() {
  int i, j;

  for (i = 0; i < N; ) {
    for (j = 0; j < N; ) {
      int r = rand();
      X[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / N;
      A[i][j] = ((DATA_TYPE)(i - 1) * (j + 4) + 2) / N;
      B[i][j] = ((DATA_TYPE)(i + 3) * (j + 7) + 3) / N;
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
        fprintf(stderr, DATA_PRINTF_MODIFIER, X[i][j]);
        if((i * N + j) % 80 == 20)
          fprintf(stderr, "\n");
      }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int t, i1, i2;
  int n = N;
  int tsteps = TSTEPS;

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
  for (t = 0; t < tsteps; t++) {
    for (i1 = 0; i1 < n; i1++)
      for (i2 = 1; i2 < n; i2++) {
        X[i1][i2] = X[i1][i2] - X[i1][i2 - 1] * A[i1][i2] / B[i1][i2 - 1];
        B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2 - 1];
      }

    for (i1 = 0; i1 < n; i1++)
      X[i1][n - 1] = X[i1][n - 1] / B[i1][n - 1];

    for (i1 = 0; i1 < n; i1++)
      for (i2 = 0; i2 < n - 2; i2++)
        X[i1][n - i2 - 2] = (X[i1][n - 2 - i2] - X[i1][n - 2 - i2 - 1]
            * A[i1][n - i2 - 3]) / B[i1][n - 3 - i2];

    for (i1 = 1; i1 < n; i1++)
      for (i2 = 0; i2 < n; i2++) {
        X[i1][i2] = X[i1][i2] - X[i1 - 1][i2] * A[i1][i2] / B[i1 - 1][i2];
        B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1 - 1][i2];
      }

    for (i2 = 0; i2 < n; i2++)
      X[n - 1][i2] = X[n - 1][i2] / B[n - 1][i2];

    for (i1 = 0; i1 < n - 2; i1++)
      for (i2 = 0; i2 < n; i2++)
        X[n - 2 - i1][i2] = (X[n - 2 - i1][i2] - X[n - i1 - 3][i2] * A[n - 3
            - i1][i2]) / B[n - 2 - i1][i2];
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
