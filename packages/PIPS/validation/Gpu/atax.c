#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>


double A[80][80];
double x[80];
double y[80];
double tmp[80];

static void init_array() {
  int i, j;

  for (i = 0; i < 80;) {
    x[i] = i * M_PI;
    for (j = 0; j < 80;) {
      A[i][j] = ((double)i * j) / 80;
      j++;
    }
    i++;
  }
}

/* Define the live-out variables. Code is not executed unless
 POLYBENCH_DUMP_ARRAYS is defined. */
static inline
void print_array(int argc, char** argv) {
  int i;
#ifndef POLYBENCH_DUMP_ARRAYS
  if(argc > 42 && !strcmp(argv[0], ""))
#endif
  {
    for (i = 0; i < 80; i++) {
      fprintf(stderr, "%0.2lf ", y[i]);
      if(i % 80 == 20)
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char** argv) {
  int i, j;

  /* Initialize array. */
  init_array();

  for (i = 0; i < 80; i++)
    y[i] = 0;
  for (i = 0; i < 80; i++) {
    tmp[i] = 0;
    for (j = 0; j < 80; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
    for (j = 0; j < 80; j++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }


  print_array(argc, argv);

  return 0;
}
