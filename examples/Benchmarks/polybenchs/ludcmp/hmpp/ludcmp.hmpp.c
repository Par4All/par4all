#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef N
# define N 1024
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

DATA_TYPE w;
DATA_TYPE a[N + 1][N + 1];
DATA_TYPE x[N + 1];
DATA_TYPE y[N + 1];
DATA_TYPE b[N + 1];

static void init_array() {
  int i, j;

  for (i = 0; i <= N;) {
    x[i] = ((DATA_TYPE)i + 1) / N;
    b[i] = ((DATA_TYPE)i + 2) / N;
    for (j = 0; j <= N;) {
      a[i][j] = ((DATA_TYPE)i * j + 1) / N;
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
    for (i = 0; i <= N; i++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, x[i]);
      if(i % 80 == 20)
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
}


#pragma hmpp myCodelet codelet, target=CUDA
void codelet(int n,
             DATA_TYPE w,
             DATA_TYPE a[N + 1][N + 1],
             DATA_TYPE x[N + 1],
             DATA_TYPE y[N + 1],
             DATA_TYPE b[N + 1]) {
  int i, j, k;
  b[0] = 1.0;
  for (i = 0; i < n; i++) {
    for (j = i + 1; j <= n; j++) {
      w = a[j][i];
      for (k = 0; k < i; k++)
        w = w - a[j][k] * a[k][i];
      a[j][i] = w / a[i][i];
    }
    for (j = i + 1; j <= n; j++) {
      w = a[i + 1][j];
      for (k = 0; k <= i; k++)
        w = w - a[i + 1][k] * a[k][j];
      a[i + 1][j] = w;
    }
  }
  y[0] = b[0];
  for (i = 1; i <= n; i++) {
    w = b[i];
    for (j = 0; j < i; j++)
      w = w - a[i][j] * y[j];
    y[i] = w;
  }
  x[n] = y[n] / a[n][n];
  for (i = 0; i <= n - 1; i++) {
    w = y[n - 1 - (i)];
    for (j = n - i; j <= n; j++)
      w = w - a[n - 1 - (i)][j] * x[j];
    x[n - 1 - (i)] = w / a[n - 1 - (i)][n - 1 - (i)];
  }
}

int main(int argc, char** argv) {
  int i, j, k;
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
  codelet(n,w,a,x,y,b);

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display(); ;

  print_array(argc, argv);

  return 0;
}
