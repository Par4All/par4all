typedef double fii[2];

typedef fii foo;

void generate04()
{
  int i = 0;
  double x = 1.;
  foo y;
  extern foo func(foo, double *);

  // use an undeclared function without source code, which returns
  // implictly an inta typedef type
  (void) func(y, &x);
}
