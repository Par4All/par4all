void dereferencing05()
{
  double * p;
  int i;

  *(p++) = 0.;
  *(p--) = 0.;
  *(++p) = 0.;
  *(--p) = 0.;
}
