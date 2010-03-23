#include <stdio.h>
#include <complex.h>

void complex03()
{
  _Complex x1 = 1 + 2I;
  float _Complex x2 = 1. + 2.I;
  double _Complex x3;
  long double _Complex x4;

  fprintf(stderr,"sizeof(x1)=%td\n", sizeof(x1));
  fprintf(stderr,"sizeof(x2)=%td\n", sizeof(x2));
  fprintf(stderr,"sizeof(x3)=%td\n", sizeof(x3));
  fprintf(stderr,"sizeof(x4)=%td\n", sizeof(x4));
}

main()
{
  complex03();
}
