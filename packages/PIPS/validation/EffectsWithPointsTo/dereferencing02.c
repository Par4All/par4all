/*
 *
 * Dimension added to p to use pointer arithmetic
 * w must be initialized...
 */

void dereferencing02()
{
  double x[2] = {1., 2.};
  double *p[1] = {&x[0]};
  double ** q = &p[0];
  double ** w;
  int i = 1;


  **q = 2.;
  **(q+(i=0)) = 3.;
  w = q;
  **(w+(i=2)) = 4.;
  x[0] = **q;
  q++;
  p[1] = *q++;
  return;
}

int main()
{
  dereferencing02();
  return 1;
}

