/* This is a partially fixed version of dereferencing02
 *
 *
 *
 */

double dereferencing13()
{
  double x[3] = {1., 2., 3.};
  double *p[3] = {&x[0], &x[1], &x[2]};
  double ** q = &p[0];
  double ** w;
  int i = 1;


  **q = 2.;
  **(q+(i=0)) = 3.;
  // w is used unitialized and this is not detected by the points-to analysis
  **(w+(i=2)) = 4.;
  x[0] = **q;
  q++;
  // *q++;
  double *z1 = *q++;
  double *z2;
  z2 = *q++;
  return *z1+*z2-**w;
}

int main()
{
  (void) dereferencing13();
  return 1;
}

