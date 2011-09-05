#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef N
# define N 4000
#endif

/* Default data type is int. */
#ifndef DATA_TYPE
# define DATA_TYPE int
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%d "
#endif

DATA_TYPE y[N][N];
DATA_TYPE sum[N][N];
DATA_TYPE beta[N];
DATA_TYPE alpha[N];
DATA_TYPE r[N]; //input
DATA_TYPE out[N]; //output

static void init_array() {
  int i;

  for (i = 0; i < N;) {
    r[i] = i * M_PI;
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
      fprintf(stderr, DATA_PRINTF_MODIFIER, r[i]);
      if(i % 80 == 20)
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
}


#pragma hmpp myCodelet codelet, target=CUDA
void codelet(int n,
             DATA_TYPE y[N][N],
             DATA_TYPE sum[N][N],
             DATA_TYPE beta[N],
             DATA_TYPE alpha[N],
             DATA_TYPE r[N],
             DATA_TYPE out[N] ) {
  int i, k;
  y[0][0] = r[0];
  beta[0] = 1;
  alpha[0] = r[0];
  for (k = 1; k < n; k++) {
    beta[k] = beta[k - 1] - alpha[k - 1] * alpha[k - 1] * beta[k - 1];
    sum[0][k] = r[k];
    for (i = 0; i <= k - 1; i++)
      sum[i + 1][k] = sum[i][k] + r[k - (i) - 1] * y[i][k - 1];
    alpha[k] = -sum[k][k] * beta[k];
    for (i = 0; i <= k - 1; i++)
      y[i][k] = y[i][k - 1] + alpha[k] * y[k - (i) - 1][k - 1];
    y[k][k] = alpha[k];
  }
  for (i = 0; i < n; i++)
    out[i] = y[i][N - 1];
}

int main(int argc, char** argv) {
  int i, k;
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
  codelet(n,y,sum,beta,alpha,r,out);


  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display();;

  print_array(argc, argv);

  return 0;
}
