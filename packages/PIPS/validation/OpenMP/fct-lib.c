// parallelization of a never called function as it might be done when
// developping in a library

typedef double t_real;

void fct_lib (const long size, t_real x[size]) {
  long   i;

  // compare loop
  for (i = 1; i < size-1; i++) {
    x[i] = x[i-1] + x[i] + x[i+1];
  }
  return;
}
