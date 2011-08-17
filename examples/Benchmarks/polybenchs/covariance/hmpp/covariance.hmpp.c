#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef M
# define M 500
#endif
#ifndef N
# define N 500
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Array declaration. */
DATA_TYPE float_n = 321414134.01;
DATA_TYPE data[M + 1][N + 1];
DATA_TYPE symmat[M + 1][M + 1];
DATA_TYPE mean[M + 1];

static void init_array() {
  int i, j;

  for (i = 0; i <= M;) {
    for (j = 0; j <= N;) {
      data[i][j] = ((DATA_TYPE)i * j) / (M + 1);
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
    for (i = 1; i <= M; i++)
      for (j = 1; j <= M; j++) {
        fprintf(stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
        if((i * M + j) % 80 == 20)
          fprintf(stderr, "\n");
      }
    fprintf(stderr, "\n");
  }
}



#pragma hmpp myCodelet codelet, target=CUDA
void codelet(int m,int n, DATA_TYPE float_n,
             DATA_TYPE data[M + 1][N + 1],
             DATA_TYPE symmat[M + 1][M + 1],
             DATA_TYPE mean[M + 1]) {
  int i, j, j1, j2;
  /* Determine mean of column vectors of input data matrix */
  for (j = 1; j <= m; j++) {
    mean[j] = 0.0;
    for (i = 1; i <= n; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }

  /* Center the column vectors. */
  for (i = 1; i <= n; i++)
    for (j = 1; j <= m; j++)
      data[i][j] -= mean[j];

  /* Calculate the m * m covariance matrix. */
  for (j1 = 1; j1 <= m; j1++)
    for (j2 = j1; j2 <= m; j2++) {
      symmat[j1][j2] = 0.0;
      for (i = 1; i <= n; i++)
        symmat[j1][j2] += data[i][j1] * data[i][j2];
      symmat[j2][j1] = symmat[j1][j2];
    }
}

int main(int argc, char** argv) {
  int i, j, j1, j2;
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

#pragma hmpp myCodelet callsite
  codelet(m,n,float_n,data,symmat,mean);


  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display();;

  print_array(argc, argv);

  return 0;
}
