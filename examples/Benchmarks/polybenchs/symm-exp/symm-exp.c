#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef Y
# define Y 512
#endif
#ifndef X
# define X 512
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
DATA_TYPE acc[X][Y];
DATA_TYPE A[Y][Y];
DATA_TYPE B[X][Y];
DATA_TYPE C[X][Y];

static void init_array() {
  int i, j;

  alpha = 12435;
  beta = 4546;
  for (i = 0; i < Y;) {
    for (j = 0; j < Y;) {
      A[i][j] = ((DATA_TYPE)i * j) / Y;
      j++;
    }
    i++;
  }
  for (i = 0; i < X;) {
    for (j = 0; j < Y;) {
      B[i][j] = ((DATA_TYPE)i * j + 1) / Y;
      C[i][j] = ((DATA_TYPE)i * j + 2) / Y;
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
    for (i = 0; i < X; i++)
      for (j = 0; j < Y; j++) {
        fprintf(stderr, DATA_PRINTF_MODIFIER, C[i][j]);
        if((i * Y + j) % 80 == 20)
          fprintf(stderr, "\n");
      }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int i, j, k;
  int n = Y;
  int m = X;

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
