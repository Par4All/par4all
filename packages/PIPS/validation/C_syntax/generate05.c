/* Use enum also */

enum count {zero,  un, deux, trois, quatre};

typedef double fii;

typedef fii foo[deux];

void generate05()
{
  int i = 0;
  double x = 1.;
  foo y;
  extern fii func(foo, double *);

  // use an undeclared function without source code, which returns
  // implictly an inta typedef type
  (void) func(y, &x);
}
