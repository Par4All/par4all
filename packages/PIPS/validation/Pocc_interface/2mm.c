#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

//#include "instrument.h"

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
#ifndef NL
# define NL 512
#endif

/* Default data type is double (dgemm). */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */DATA_TYPE
    alpha1;
DATA_TYPE beta1;
DATA_TYPE alpha2;
DATA_TYPE beta2;

DATA_TYPE C[NI][NJ];
DATA_TYPE A[NI][NK];
DATA_TYPE B[NK][NJ];
DATA_TYPE D[NJ][NL];
DATA_TYPE E[NI][NL];

static inline
void init_array() {
  int i, j;

  alpha1 = 32412;
  beta1 = 2123;
  alpha2 = 132412;
  beta2 = 92123;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++) {
      int r = rand();
      A[i][j] = ((DATA_TYPE)i * j) / NI;
    }
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++) {
      int r = rand();
      B[i][j] = ((DATA_TYPE)i * j + 1) / NJ;
    }
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++) {
      int r = rand();
      C[i][j] = ((DATA_TYPE)i * j + 2) / NJ;
    }
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++) {
      int r = rand();
      D[i][j] = ((DATA_TYPE)i * j + 2) / NJ;
    }
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++) {
      int r = rand();
      E[i][j] = ((DATA_TYPE)i * j + 2) / NJ;
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
    for (i = 0; i < NI; i++) {
      for (j = 0; j < NL; j++) {
        fprintf(stderr, "%0.2lf ", E[i][j]);
        if((i * NI + j) % 80 == 20)
          fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
  }
}

int main(int argc, char** argv) {
  int i, j, k;
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  //polybench_start_instruments;

  /* Define the live-in variables. Code is never executed */
  if(argc > 42 && !strcmp(argv[0], ""))
    init_array();

  /* E := A*B*D */
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      C[i][j] = 0;
      for (k = 0; k < nk; ++k)
        C[i][j] += A[i][k] * B[k][j];
    }
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      E[i][j] = 0;
      for (k = 0; k < nj; ++k)
        E[i][j] += C[i][k] * D[k][j];
    }

  /* Define the live-out variables. Code is never executed */
  if(argc > 42 && !strcmp(argv[0], ""))
    print_array(argc, argv);

  /* Stop and print timer. */
  //polybench_stop_instruments; polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
