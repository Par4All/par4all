// array of pointers towards arrays

void foo(double * (*t[3][4])[5][6][7])
{
  *(*(t[2][3]))[1][2][3] = 2.5;
  return;
}

void pointer14()
{
  double * (*t[3][4])[5][6][7];
  double * a[5][6][7];
  double z;
  t[1][2] = &a;
  a[1][2][3] = &z;
  foo(t);
  return;
}
