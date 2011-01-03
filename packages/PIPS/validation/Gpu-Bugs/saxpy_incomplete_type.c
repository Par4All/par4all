#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
//#include "types.h"

#ifdef USE_DOUBLE
typedef double t_precision;
#else
typedef float t_precision;
#endif

void help_mesage (void) {
  printf ("usage: ./saxpy vec_size\n");
  exit (1);
}

void saxpy (long long size, t_precision dst[], t_precision src[], t_precision alpha) {
  long long i = 0;
  for (i = 0; i < size; i++) dst[i] = alpha * src[i] + dst[i];
}

void init (long long size, t_precision ptr[], t_precision val) {
  long long i = 0;
  for (i = 0; i < size; i++) ptr[i] = val;
}

t_precision sum ( long long size, t_precision ptr[]) {
  long long i = 0;
  t_precision result =0;
  for (i = 0; i < size; i++) result += ptr[i];
  return result;
}

long long get_vec_size (int argc, char** argv) {
  long long result = 0;
  if (argc != 2) help_mesage ();
  result = (long long) atol (argv[1]);
  return result;
}

int main (int argc, char** argv) {
  long long size = get_vec_size (argc, argv);
  t_precision* x = (t_precision*) malloc (sizeof (t_precision) * size);
  t_precision* y = (t_precision*) malloc (sizeof (t_precision) * size);
  init (size, x, 1.0);
  init (size, y, 0.0);

  saxpy (size, y, x, 1.0);

  printf ("sum = %.5f\n", sum(size, y));
  return 0;
}
