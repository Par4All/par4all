#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef TSTEPS
# define TSTEPS 10000
#endif
#ifndef N
# define N 4096*10
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

DATA_TYPE A[N];
DATA_TYPE B[N];

static void init_array() {
  int i, j;

  for (i = 0; i < N;) {
    A[i] = ((DATA_TYPE)4 * i + 10) / N;
    B[i] = ((DATA_TYPE)7 * i + 11) / N;
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
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i]);
      if(i % 80 == 20)
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int t, i, j;
  int tsteps = TSTEPS;
  int n = N;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  timer_start();

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    init_array();
  }

  for (t = 0; t < tsteps; t++) {
    for (i = 2; i < n - 1; i++)
      B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);

    for (j = 2; j < n - 1; j++)
      A[j] = B[j];
  }

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display();;

  print_array(argc, argv);

  return 0;
}
