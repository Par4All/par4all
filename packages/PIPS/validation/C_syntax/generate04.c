typedef double fii[2];

typedef fii foo;

typedef double faa;

void generate04()
{
  int i = 0;
  double x = 1.;
  foo y;
  extern faa func(foo, double *);

  // use an undeclared function without source code, which returns
  // implictly an inta typedef type
  (void) func(y, &x);
}
