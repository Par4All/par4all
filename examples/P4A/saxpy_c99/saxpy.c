#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>

#ifdef USE_DOUBLE
typedef double t_precision;
#else
typedef float t_precision;
#endif

void help_mesage (void) {
  printf ("usage: ./saxpy vec_size\n");
  exit (1);
}

void saxpy (long size, t_precision z[size], t_precision x[size], t_precision y[size], t_precision alpha) {
  long i = 0;
  for (i = 0; i < size; i++) z[i] = alpha * x[i] + y[i];
}

void init (long size, t_precision ptr[size], t_precision val) {
  long i = 0;
  for (i = 0; i < size; i++) ptr[i] = val;
}

bool check_saxpy (long size, t_precision ptr[size],
				  t_precision alpha, t_precision x, t_precision y) {
  bool result = true;
  long i = 0;
  t_precision saxpy = alpha * x + y;
  for (i = 0; i < size; i++) result &= (ptr[i] == saxpy);
  return result;
}

long get_vec_size (int argc, char** argv) {
  long result = 0;
  if (argc != 2) help_mesage ();
  result = (long) atol (argv[1]);
  return result;
}

int main (int argc, char** argv) {
  long size = get_vec_size (argc, argv);
  t_precision a0 = 1.0;
  t_precision x0 = 1.0;
  t_precision y0 = 0.0;
  t_precision x[size];
  t_precision y[size];
  t_precision z[size];
  init (size, x, x0);
  init (size, y, y0);

  saxpy (size, z, x, y, a0);

  if (check_saxpy (size, z, a0, x0, y0) == false) {
	printf ("check failed\n");
	return 1;
  }

  printf ("check succeded\n");

  return 0;
}
