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

DATA_TYPE A[N][N];
DATA_TYPE x1[N];
DATA_TYPE y_1[N];
DATA_TYPE x2[N];
DATA_TYPE y_2[N];

static void init_array() {
  int i, j;

  for (i = 0; i < N;) {
    x1[i] = ((DATA_TYPE)i) / N;
    x2[i] = ((DATA_TYPE)i + 1) / N;
    y_1[i] = ((DATA_TYPE)i + 3) / N;
    y_2[i] = ((DATA_TYPE)i + 4) / N;
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
      fprintf(stderr, "%0.2lf ", x1[i]);
      fprintf(stderr, "%0.2lf ", x2[i]);
      if((2 * i) % 80 == 20)
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
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

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display(); ;

  print_array(argc, argv);

  return 0;
}
