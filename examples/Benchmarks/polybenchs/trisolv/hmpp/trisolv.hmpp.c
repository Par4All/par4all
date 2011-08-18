#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef N
# define N 4000
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
DATA_TYPE A[N][N];
DATA_TYPE x[N];
DATA_TYPE c[N];

static void init_array() {
  int i, j;

  for (i = 0; i < N;) {
    c[i] = ((DATA_TYPE)i) / N;
    for (j = 0; j < N;) {
      A[i][j] = ((DATA_TYPE)i * j) / N;
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
    for (i = 0; i < N; i++) {
      fprintf(stderr, "%0.2lf ", x[i]);
      if((2 * i) % 80 == 20)
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
}

#pragma hmpp myCodelet codelet, target=CUDA
void codelet(int n,
             DATA_TYPE A[N][N],
             DATA_TYPE x[N],
             DATA_TYPE c[N]) {
  int i, j;
  for (i = 0; i < n; i++) {
    x[i] = c[i];
    for (j = 0; j <= i - 1; j++)
      x[i] = x[i] - A[i][j] * x[j];
    x[i] = x[i] / A[i][i];
  }
}
int main(int argc, char** argv) {
  int i, j;
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
  codelet(n,A,x,c);

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }


  /* Stop and print timer. */
  timer_stop_display(); ;

  print_array(argc, argv);

  return 0;
}
