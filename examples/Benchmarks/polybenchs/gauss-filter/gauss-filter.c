#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "timing.h"

/* Default problem size. */
#ifndef X
# define X 1920
#endif
#ifndef Y
# define Y 1080
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
DATA_TYPE g_tmp_image[Y][X];
DATA_TYPE g_acc1[Y][X][4];
DATA_TYPE g_acc2[Y][X][4];
DATA_TYPE in_image[Y][X]; //input
DATA_TYPE gauss_image[Y][X]; //output

static void init_array() {
  int i, j;

  for (i = 0; i < Y;) {
    for (j = 0; j < X;) {
      in_image[i][j] = ((DATA_TYPE)i * j) / X;
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
static void print_array(int argc, char** argv) {
  int i, j;
#ifndef POLYBENCH_DUMP_ARRAYS
  if(argc > 42 && !strcmp(argv[0], ""))
#endif
  {
    for (i = 1; i < Y-1; i++)
      for (j = 1; j < X-1; j++) {
        fprintf(stderr, DATA_PRINTF_MODIFIER, gauss_image[i][j]);
        if((i * Y + j) % 80 == 20)
          fprintf(stderr, "\n");
      }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int x, y, k;
  int t = T;
  int m = X;
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
#ifdef PGCC
#pragma scop
#endif
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
#ifdef PGCC
#pragma endscop
#endif
#ifdef PGI_ACC
}
#endif

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[0]==0) {
    print_array(argc, argv);
  }

  /* Stop and print timer. */
  timer_stop_display();;

  print_array(argc, argv);

  return 0;
}
