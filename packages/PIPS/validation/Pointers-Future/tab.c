#include <stdio.h>
#include <stdlib.h>

typedef double*restrict t_rdp;

// sum pointer 
static double tsump (const long size, t_rdp* tab) {
  long i;
  long j;
  long result = 0.0;

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      result += tab[i][j];
    }
  }
  return result;
}

// sum array linear
static double tsuml (const long size, double* tab) {
  long i;
  long j;
  long result = 0.0;

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      result += tab[i*size+j];
    }
  }
  return result;
}

// sum array C99
double tsumc (const long size, double tab[size][size]) {
  return tsuml (size, &tab[0][0]);
}

// initialization pointer
static void tinitp (const long size, t_rdp* tab) {
  long i;
  long j;

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      tab[i][j] = 1.0;
    }
  }
}

// initialization linear
static void tinitl(const long size, double* tab) {
  long i;
  long j;

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      tab[i*size+j] = 1.0;
    }
  }
}

// initialization C99
void tinitc(const long size, double tab[size][size]) {
  tinitl (size, &tab[0][0]);
}

int main (const int argc, const char** argv) {
  long i;
  long size  = atol (argv[1]);
  double sum = 0.0;

  // allocation
  double  dbc[size][size];
  t_rdp*  dbp = (t_rdp*)   malloc (size * sizeof (t_rdp));
  double* dbl = (double*)  malloc (size * size * sizeof (double));
  for (i = 0; i < size; i++)
    dbp[i] = (t_rdp) malloc (size * sizeof (double));

  // initialization
  tinitp (size, dbp);
  tinitl (size, dbl);
  tinitc (size, dbc);

  // method pointer
  sum = tsump (size, dbp);
  printf ("sum pointer: %lf\n", sum);

  // method linear
  sum = tsuml (size, dbl);
  printf ("sum linear : %lf\n", sum);

  // method c99
  sum = tsumc (size, dbc);
  printf ("sum C99    : %lf\n", sum);

  // cleanup
  for (i = 0; i < size; i++) free ((void*) dbp[i]);
  free ((void*) dbp);
  free ((void*) dbl);
  return 0;
}
