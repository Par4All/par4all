#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef Y
# define Y 1024
#endif
#ifndef X
# define X 1024
#endif

/* Default data type is double (dsyrk). */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

DATA_TYPE alpha;
DATA_TYPE beta;
DATA_TYPE A[Y][X];
DATA_TYPE C[Y][Y];

static void init_array() {
  int i, j;

  alpha = 12435;
  beta = 4546;
  for (i = 0; i < Y;) {
    for (j = 0; j < X;) {
      A[i][j] = ((DATA_TYPE)i * j) / Y;
      j++;
    }
    for (j = 0; j < Y; ) {
      C[i][j] = ((DATA_TYPE)i * j + 2) / Y;
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
    for (i = 0; i < Y; i++)
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
  /*  C := alpha*A*A' + beta*C */
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] *= beta;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < m; k++)
        C[i][j] += alpha * A[i][k] * A[j][k];
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
  timer_stop_display();;

  print_array(argc, argv);

  return 0;
}
