#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef NR
# define NR 128
#endif
#ifndef NQ
# define NQ 128
#endif
#ifndef NP
# define NP 128
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

DATA_TYPE A[NR][NQ][NP];
DATA_TYPE sum[NR][NQ][NP];
DATA_TYPE C4[NP][NP];

static void init_array() {
  int i, j, k;

  for (i = 0; i < NR;) {
    for (j = 0; j < NQ;) {
      for (k = 0; k < NP;) {
        A[i][j][k] = ((DATA_TYPE)i * j + k) / NP;
        k++;
      }
      j++;
    }
    i++;
  }
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      C4[i][j] = ((DATA_TYPE)i * j) / NP;
      j++;
    }
    i++;
  }
}

/* Define the live-out variables. Code is not executed unless
 POLYBENCH_DUMP_ARRAYS is defined. */
static inline
void print_array(int argc, char** argv) {
  int i, j, k;
#ifndef POLYBENCH_DUMP_ARRAYS
  if(argc > 42 && !strcmp(argv[0], ""))
#endif
  {
    for (i = 0; i < NR; i++)
      for (j = 0; j < NQ; j++)
        for (k = 0; k < NP; k++) {
          fprintf(stderr, "%0.2lf ", A[i][j][k]);
          if((i * NR + j * NQ + k) % 80 == 20)
            fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
  }
}


#pragma hmpp myCodelet codelet, target=CUDA
void codelet(int nr,int nq,int np,
             DATA_TYPE A[NR][NQ][NP],
             DATA_TYPE sum[NR][NQ][NP],
             DATA_TYPE C4[NP][NP]) {
  int r, q, p, s;
  for (r = 0; r < nr; r++)
    for (q = 0; q < nq; q++) {
      for (p = 0; p < np; p++) {
        sum[r][q][p] = 0;
        for (s = 0; s < np; s++)
          sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < np; p++)
        A[r][q][p] = sum[r][q][p];
    }
}

int main(int argc, char** argv) {
  int r, q, p, s;
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  timer_start();

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    init_array();
  }

#pragma hmpp myCodelet callsite
  codelet(nr,nq,np,A,sum,C4);


  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display();;

  print_array(argc, argv);

  return 0;
}
