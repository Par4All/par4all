void point_to01()
{
  double x = 1.;
  double * p = &x;
  double * q = 0;

  q = p;
  *p = 2.;
}
