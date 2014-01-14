/* Excerpt of dereferencing16.c to study the bug with *q++
 *
 */

double dereferencing17()
{
  double x[3] = {1., 2., 3.};
  double *p[3] = {&x[0], &x[1], &x[2]};
  double ** q = &p[0];

  q++;
  double *z1 = *q++;
  return *z1;
}

int main()
{
  (void) dereferencing17();
  return 1;
}
