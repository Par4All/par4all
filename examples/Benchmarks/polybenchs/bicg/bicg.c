#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef NX
# define NX 8000
#endif
#ifndef NY
# define NY 8000
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

DATA_TYPE A[NX][NY];
DATA_TYPE r[NX];
DATA_TYPE s[NX];
DATA_TYPE p[NX];
DATA_TYPE q[NX];

static void init_array() {
  int i, j;

  for (i = 0; i < NX;i++) {
    r[i] = i * M_PI;
    p[i] = i * M_PI;
    for (j = 0; j < NY;j++) {
      A[i][j] = ((DATA_TYPE)i * j) / NX;
    }
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
    for (i = 0; i < NX; i++) {
      fprintf(stderr, "%0.2lf ", s[i]);
      fprintf(stderr, "%0.2lf ", q[i]);
      if(i % 80 == 20)
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int i, j;
  int nx = NX;
  int ny = NY;

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
  for (i = 0; i < ny; i++)
    s[i] = 0;

  for (i = 0; i < nx; i++) {
    q[i] = 0;
    for (j = 0; j < ny; j++) {
      s[j] = s[j] + r[i] * A[i][j];
      q[i] = q[i] + A[i][j] * p[j];
    }
  }
#ifdef PGI_ACC
}
#endif


  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display(); ;

  print_array(argc, argv);

  return 0;
}
