#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef Y
# define Y 4000
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

DATA_TYPE alpha;
DATA_TYPE beta;
DATA_TYPE A[Y][Y];
DATA_TYPE B[Y][Y];
DATA_TYPE x[Y];
DATA_TYPE u1[Y];
DATA_TYPE u2[Y];
DATA_TYPE v2[Y];
DATA_TYPE v1[Y];
DATA_TYPE w[Y];
DATA_TYPE y[Y];
DATA_TYPE z[Y];

static void init_array() {
  int i, j;

  alpha = 43532;
  beta = 12313;
  for (i = 0; i < Y;) {
    u1[i] = i;
    u2[i] = (i + 1) / Y / 2.0;
    v1[i] = (i + 1) / Y / 4.0;
    v2[i] = (i + 1) / Y / 6.0;
    y[i] = (i + 1) / Y / 8.0;
    z[i] = (i + 1) / Y / 9.0;
    x[i] = 0.0;
    w[i] = 0.0;
    for (j = 0; j < Y;) {
      A[i][j] = ((DATA_TYPE)i * j) / Y;
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
    for (i = 0; i < Y; i++) {
      fprintf(stderr, "%0.2lf ", w[i]);
      if(i % 80 == 20)
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int i, j;
  int n = Y;

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
  for (i = 0; i < Y; i++)
    for (j = 0; j < Y; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < Y; i++)
    for (j = 0; j < Y; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < Y; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < Y; i++)
    for (j = 0; j < Y; j++)
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
