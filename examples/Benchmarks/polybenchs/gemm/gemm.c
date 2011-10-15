#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef NI
# define NI 512
#endif
#ifndef NJ
# define NJ 512
#endif
#ifndef NK
# define NK 512
#endif

/* Default data type is double (dgemm). */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

DATA_TYPE alpha;
DATA_TYPE beta;
DATA_TYPE C[NI][NJ];
DATA_TYPE A[NI][NK];
DATA_TYPE B[NK][NJ];

static void init_array() {
  int i, j;

  alpha = 32412;
  beta = 2123;
  for (i = 0; i < NI; ) {
    for (j = 0; j < NK; ) {
      A[i][j] = ((DATA_TYPE)i * j) / NI;
      j++;
    }
    i++;
  }
  for (i = 0; i < NK; ) {
    for (j = 0; j < NJ; ) {
      B[i][j] = ((DATA_TYPE)i * j + 1) / NJ;
      j++;
    }
    i++;
  }
  for (i = 0; i < NI; ) {
    for (j = 0; j < NJ; ) {
      C[i][j] = ((DATA_TYPE)i * j + 2) / NJ;
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
    for (i = 0; i < NI;) {
      for (j = 0; j < NJ;) {
        fprintf(stderr, "%0.2lf ", C[i][j]);
        if((i * NI + j) % 80 == 20)
          fprintf(stderr, "\n");
        j++;
      }
      fprintf(stderr, "\n");
      i++;
    }
  }
}

int main(int argc, char** argv) {
  int i, j, k;
  int ni = NI;
  int nj = NJ;
  int nk = NK;

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
  /* C := alpha*A*B + beta*C */
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      C[i][j] *= beta;
      for (k = 0; k < nk; ++k)
        C[i][j] += alpha * A[i][k] * B[k][j];
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
  timer_stop_display();;

  print_array(argc, argv);

  return 0;
}
