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
#ifndef NL
# define NL 512
#endif
#ifndef NM
# define NM 512
#endif

/* Default data type is double (dgemm). */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

DATA_TYPE A[NI][NK];
DATA_TYPE B[NK][NJ];
DATA_TYPE C[NJ][NM];
DATA_TYPE D[NM][NL];
DATA_TYPE E[NI][NJ];
DATA_TYPE F[NJ][NL];
DATA_TYPE G[NI][NL];

static void init_array() {
  int i, j;

  for (i = 0; i < NI;) {
    for (j = 0; j < NK;) {
      int r = rand();
      A[i][j] = ((DATA_TYPE)i * j) / NI;
      j++;
    }
    i++;
  }
  for (i = 0; i < NK;) {
    for (j = 0; j < NJ;) {
      int r = rand();
      B[i][j] = ((DATA_TYPE)i * j + 1) / NJ;
      j++;
    }
    i++;
  }
  for (i = 0; i < NJ;) {
    for (j = 0; j < NM;) {
      int r = rand();
      C[i][j] = ((DATA_TYPE)i * j + 2) / NJ;
      j++;
    }
    i++;
  }
  for (i = 0; i < NM;) {
    for (j = 0; j < NL;) {
      int r = rand();
      D[i][j] = ((DATA_TYPE)i * j + 2) / NJ;
      j++;
    }
    i++;
  }
  for (i = 0; i < NI;) {
    for (j = 0; j < NJ;) {
      int r = rand();
      E[i][j] = ((DATA_TYPE)i * j + 2) / NJ;
      j++;
    }
    i++;
  }
  for (i = 0; i < NJ;) {
    for (j = 0; j < NL;) {
      int r = rand();
      F[i][j] = ((DATA_TYPE)i * j + 2) / NJ;
      j++;
    }
    i++;
  }
  for (i = 0; i < NI;) {
    for (j = 0; j < NL;) {
      int r = rand();
      G[i][j] = ((DATA_TYPE)i * j + 2) / NJ;
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
    for (i = 0; i < NI; i++) {
      for (j = 0; j < NL; j++) {
        fprintf(stderr, "%0.2lf ", G[i][j]);
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
  int nm = NM;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  timer_start();

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0] == 0) {
    init_array();
  }

#ifdef PGI_ACC
#pragma acc region
{
#endif
  /* E := A*B */
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      E[i][j] = 0;
      for (k = 0; k < nk; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }

  /* F := C*D */
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++) {
      F[i][j] = 0;
      for (k = 0; k < nm; ++k)
        F[i][j] += C[i][k] * D[k][j];
    }
  /* G := E*F */
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      G[i][j] = 0;
      for (k = 0; k < nj; ++k)
        G[i][j] += E[i][k] * F[k][j];
    }
#ifdef PGI_ACC
}
#endif

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0] == 0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display();
  ;

  print_array(argc, argv);

  return 0;
}
