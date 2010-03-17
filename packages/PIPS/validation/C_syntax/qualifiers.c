
void h(int n, int * const volatile p, int * volatile * const q, int * const r)
{
  volatile int *const ptr;
  const int ar[10][20];
  int const b[10];
  const volatile int i;
  int const volatile j;
  const struct s {int mem;} cs = {1};
  struct s ncs;
  typedef int A[2][3];
  const A a = {{4,5,6},{7,8,9}};
  int *pi;
  const int *pci;
}
