typedef double mydouble;

struct s {
  int dim;
};

void typedef04(int n, struct s d, struct s * pd)
{
  typedef int narray_t[n+2];
  typedef int darray_t[d.dim];
  typedef int parray_t[pd->dim];
  narray_t a1;
  darray_t a2;
}
