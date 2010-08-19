// parallelization of a never called function
// in such a case pips paralelized the function Even if the parallelization
// is uselees the output code should compile

typedef double t_real;

t_real dotp_ptr (const long size, const t_real* x) {
  long   i;
  t_real result = 0.0;

  // compare loop
  for (i = 0; i < size; i++) {
    result += x[i] * x[i];
  }
  return result;
}


int main () {
  long    i;
  t_real  v[100];

  // load vector
  for (i = 0; i < 100; i++) v[i] = 1.0;

  // exit without doing anything
  return 0;
}
