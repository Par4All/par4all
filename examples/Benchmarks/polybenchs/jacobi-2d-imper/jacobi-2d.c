#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef TSTEPS
# define TSTEPS 20
#endif
#ifndef Y
# define Y 1000
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
DATA_TYPE A[Y][Y];
DATA_TYPE B[Y][Y];

static void init_array() {
  int i, j;

  for (i = 0; i < Y;) {
    for (j = 0; j < Y;) {
      A[i][j] = ((DATA_TYPE)i * (j + 2) + 10) / Y;
      B[i][j] = ((DATA_TYPE)(i - 4) * (j - 1) + 11) / Y;
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
    for (i = 0; i < Y; i++)
      for (j = 0; j < Y; j++) {
        fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
        if((i * Y + j) % 80 == 20)
          fprintf(stderr, "\n");
      }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int t, i, j;
  int tsteps = TSTEPS;
  int n = Y;

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
  for (t = 0; t < tsteps; t++) {
    for (i = 2; i < n - 1; i++)
      for (j = 2; j < n - 1; j++)
        B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j]
            + A[i - 1][j]);
    for (i = 2; i < n - 1; i++)
      for (j = 2; j < n - 1; j++)
        A[i][j] = B[i][j];
  }
#ifdef PGI_ACC
}
#endif

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display();

  print_array(argc, argv);

  return 0;
}
