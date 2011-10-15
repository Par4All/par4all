#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef N
# define N 4000
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

DATA_TYPE alpha;
DATA_TYPE beta;
DATA_TYPE A[N][N];
DATA_TYPE B[N][N];
DATA_TYPE x[N];
DATA_TYPE u1[N];
DATA_TYPE u2[N];
DATA_TYPE v2[N];
DATA_TYPE v1[N];
DATA_TYPE w[N];
DATA_TYPE y[N];
DATA_TYPE z[N];

static void init_array() {
  int i, j;

  alpha = 43532;
  beta = 12313;
  for (i = 0; i < N;) {
    u1[i] = i;
    u2[i] = (i + 1) / N / 2.0;
    v1[i] = (i + 1) / N / 4.0;
    v2[i] = (i + 1) / N / 6.0;
    y[i] = (i + 1) / N / 8.0;
    z[i] = (i + 1) / N / 9.0;
    x[i] = 0.0;
    w[i] = 0.0;
    for (j = 0; j < N;) {
      A[i][j] = ((DATA_TYPE)i * j) / N;
      j++;
    }
    i++;
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
    for (i = 0; i < N; i++) {
      fprintf(stderr, "%0.2lf ", w[i]);
      if(i % 80 == 20)
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int i, j;
  int n = N;

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
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      w[i] = w[i] + alpha * A[i][j] * x[j];
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
