// Check that formal parameters and global variables are proprely
// transformed or not transformed

int c[3];

void constant_array_scalarization02(int a[3])
{
  int b[3];

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
