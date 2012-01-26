void dereferencing02()
{
  double x = 1.;
  double *p = &x;
  double ** q = &p;
  double ** w;
  int i = 1;


  **q = 2.;
  **(q+(i=0)) = 3.;
  **(w+(i=2)) = 4.;
  x = **q;
  q++;
  *q++;
}

int main()
{
  dereferencing02();
  return 1;
}

