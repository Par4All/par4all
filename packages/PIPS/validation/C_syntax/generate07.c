/* Use struct, union and bit also */

enum count {zero,  un, deux, trois, quatre};

enum us_count {one=un, two, three, four};

enum kount {null, eins, zwei, drei, vier};

typedef struct {
  int bonjour:two+drei;
  char hello:three;
} morning_t;

typedef double fii;

typedef fii foo[deux];

void generate07()
{
  int i = 0;
  double x = 1.;
  foo y;
  typedef union {
    int either;
    double or;
  } z_t;
  z_t z;
  morning_t u;
  extern fii func(foo, z_t, morning_t, double *);

  // use an undeclared function without source code, which returns
  // implictly an inta typedef type
  (void) func(y, z, u, &x);
}
