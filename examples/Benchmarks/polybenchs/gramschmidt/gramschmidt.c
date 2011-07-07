#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef M
# define M 512
#endif
#ifndef N
# define N 512
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

DATA_TYPE nrm;
DATA_TYPE A[M][N];
DATA_TYPE R[M][N];
DATA_TYPE Q[M][N];

static void init_array() {
  int i, j;

  for (i = 0; i < M; ) {
    for (j = 0; j < N; ) {
      A[i][j] = ((DATA_TYPE)(i + 1) * (j + 1)) / M;
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
    for (i = 0; i < M; i++)
      for (j = 0; j < N; j++) {
        fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
        if((i * M + j) % 80 == 20)
          fprintf(stderr, "\n");
      }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int i, j, k;
  int m = M;
  int n = N;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  timer_start();

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    init_array();
  }

  for (k = 0; k < n; k++) {
    for(j=0;j<1;j++) {
      nrm = 0;
      for (i = 0; i < m; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = sqrt(nrm);
    }
    for (i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];
    for (j = k + 1; j < n; j++) {
      R[k][j] = 0;
      for (i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (i = 0; i < m; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display(); ;

  print_array(argc, argv);

  return 0;
}
