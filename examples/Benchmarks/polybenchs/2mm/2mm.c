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


double alpha1;
double beta1;
double alpha2;
double beta2;
double C[NI][NJ];
double A[NI][NK];
double B[NK][NJ];
double D[NJ][NL];
double E[NI][NL];

static void init_array() {
  int i, j;

  alpha1 = 32412;
  beta1 = 2123;
  alpha2 = 132412;
  beta2 = 92123;
  for (i = 0; i < NI; ) {
    for (j = 0; j < NK; ) {
      A[i][j] = ((double)i * j) / NI;
      j++;
    }
    i++;
  }
  for (i = 0; i < NK; ) {
    for (j = 0; j < NJ; ) {
      B[i][j] = ((double)i * j + 1) / NJ;
      j++;
    }
    i++;
  }
  for (i = 0; i < NI; ) {
    for (j = 0; j < NJ; ) {
      C[i][j] = ((double)i * j + 2) / NJ;
      j++;
    }
    i++;
  }
  for (i = 0; i < NJ; ) {
    for (j = 0; j < NL; ) {
      D[i][j] = ((double)i * j + 2) / NJ;
      j++;
    }
    i++;
  }
  for (i = 0; i < NI; ) {
    for (j = 0; j < NL; ) {
      E[i][j] = ((double)i * j + 2) / NJ;
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
  timer_start();

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0] == 0) {
    init_array();
  }

  /* E := A*B*D */
#ifdef PGI_ACC
#pragma acc region
{
#endif
#ifdef PGCC
#pragma scop
#endif
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
#ifdef PGCC
#pragma endscop
#endif
#ifdef PGI_ACC
}
#endif

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0] == 0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display();
  
  print_array(argc, argv);

  return 0;
}
