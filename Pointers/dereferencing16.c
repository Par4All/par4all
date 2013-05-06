/* Excerpt of dereferencing13.c where the analysis is stopped by an
 * illegal dereferencement of w (see dereferencing15).
 *
 * Here, w and its references have been removed.
 *
 */

double dereferencing16()
{
  double x[3] = {1., 2., 3.};
  double *p[3] = {&x[0], &x[1], &x[2]};
  double ** q = &p[0];
  int i = 1;


  **q = 2.;
  **(q+(i=0)) = 3.;
  x[0] = **q;
  q++;
  // *q++;
  double *z1 = *q++;
  double *z2;
  z2 = *q++;
  return *z1+*z2;
}

int main()
{
  (void) dereferencing16();
  return 1;
}

