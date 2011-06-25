#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef M
# define M 1920
#endif
#ifndef N
# define N 1080
#endif
#ifndef T
# define T 1920
#endif

/* Default data type is int. */
#ifndef DATA_TYPE
# define DATA_TYPE int
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%d "
#endif

DATA_TYPE tot[4];
DATA_TYPE Gauss[4];
DATA_TYPE g_tmp_image[N][M];
DATA_TYPE g_acc1[N][M][4];
DATA_TYPE g_acc2[N][M][4];
DATA_TYPE in_image[N][M]; //input
DATA_TYPE gauss_image[N][M]; //output

static inline
void init_array() {
  int i, j;

  for (i = 0; i < N;) {
    for (j = 0; j < M;) {
      in_image[i][j] = ((DATA_TYPE)i * j) / M;
      j++;
    }
    i++;
  }
  for (i = 0; i < 4;) {
    Gauss[i] = i;
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
    for (i = 1; i < N-1; i++)
      for (j = 1; j < M-1; j++) {
        fprintf(stderr, DATA_PRINTF_MODIFIER, gauss_image[i][j]);
        if((i * N + j) % 80 == 20)
          fprintf(stderr, "\n");
      }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int x, y, k;
  int t = T;
  int m = M;
  int n = N;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  timer_start();

  tot[0] = 0;
  for (k = t - 1; k <= 1 + t; k++)
    tot[k + 2 - t] = tot[k + 1 - t] + Gauss[k - t + 1];
  for (k = t - 1; k <= 1 + t; k++)
    tot[k + 2 - t] = tot[k + 1 - t] + Gauss[k - t + 1];
  for (x = 1; x < n - 2; x++) {
    for (y = 0; y < m; y++) {
      g_acc1[x][y][0] = 0;
      for (k = t - 1; k <= 1 + t; k++)
        g_acc1[x][y][k + 2 - t] = g_acc1[x][y][k + 1 - t]
            + in_image[x + k - t][y] * Gauss[k - t + 1];
      g_tmp_image[x][y] = g_acc1[x][y][3] / tot[3];
    }
  }
  for (x = 1; x < n - 1; x++) {
    for (y = 1; y < m - 1; y++) {
      g_acc2[x][y][0] = 0;
      for (k = t - 1; k <= 1 + t; k++)
        g_acc2[x][y][k + 2 - t] = g_acc2[x][y][k + 1 - t] + g_tmp_image[x][y
            + k - t] * Gauss[k - t + 1];
      gauss_image[x][y] = g_acc2[x][y][3] / tot[3];
    }
  }

  /* Stop and print timer. */
  timer_stop_display();;

  print_array(argc, argv);

  return 0;
}
