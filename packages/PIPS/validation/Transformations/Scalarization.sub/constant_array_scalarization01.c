// Check that the new variables are declared in alphabetical order

void constant_array_scalarization01()
{
  int a[3], b[3], c[3];

  a[0] = 1;
  a[1] = 1;
  a[2] = 1;

  c[0] = 1;
  c[1] = 1;
  c[2] = 1;

  b[2] = 1;
  b[1] = 1;
  b[0] = 1;
}
