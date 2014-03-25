// See what happens with different types

void constant_array_scalarization06()
{
  int a[3];
  double b[3];
  long long int c[3];

  a[0] = 1;
  a[1] = 1;
  a[2] = 1;

  c[0] = 1L;
  c[1] = 1L;
  c[2] = 1L;

  b[2] = 1.;
  b[1] = 1.;
  b[0] = 1.;
}
