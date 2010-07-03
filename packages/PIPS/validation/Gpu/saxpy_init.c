#ifdef USE_DOUBLE
typedef double t_precision;
#else
typedef float t_precision;
#endif

enum {size = 1000000 };

void init (long long size, t_precision ptr[size], t_precision val) {
  long long i = 0;
  for (i = 0; i < size; i++) ptr[i] = val;
}

int main (int argc, char** argv) {
  t_precision* x;
  t_precision* y;
  init (size, x, 1.0);
  init (size, y, 0.0);

  return 0;
}
