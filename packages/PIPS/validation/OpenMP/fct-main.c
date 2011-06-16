// parallelization of a never called function as it might be done when
// developping in a library. In such a case PIPS parallelizes the never
// called function.
// Behavior changed from r19798: the loop is now conservatively declared as
// sequential which seems better, since it is actually sequential in the general
// case (BC).

typedef double t_real;

void fct_lib (const long size, t_real x[size]) {
  long   i;

  // compare loop
  for (i = 1; i < size-1; i++) {
    x[i] = x[i-1] + x[i] + x[i+1];
  }
  return;
}

int main () {
  long    i;
  t_real  v[100];

  // load vector
  for (i = 0; i < 100; i++) v[i] = 1.0;

  // exit without doing anything
  return 0;
}
