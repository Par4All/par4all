void dereferencing02()
{
  double x = 1.;
  double * p = &x;
  double ** q = &p;
  int i = 1;


  **q = 2.;
  **(q+(i=2)) = 3.;
  x = **q;
  q++;
  *q++;
}
