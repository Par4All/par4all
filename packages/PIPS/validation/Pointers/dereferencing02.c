double dereferencing02()
{
  double x = 1.;
  double *p = &x;
  double ** q = &p;
  double ** w;
  int i = 1;


  **q = 2.;
  **(q+(i=0)) = 3.;
  // w is used unitialized and this is not detected by the points-to analysis
  **(w+(i=2)) = 4.;
  x = **q;
  // Since p is not an array, this incrementation is undefined
  q++;
  // *q++;
  double *z1 = *q++;
  double *z2;
  z2 = *q++;
  return *z2-**w;
}

int main()
{
  dereferencing02();
  return 1;
}

