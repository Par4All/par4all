/* Use struct, union and bit also. Does not work if the union is not named. */

enum count {zero,  un, deux, trois, quatre};

enum us_count {one=un, two, three, four};

typedef struct {
  int bonjour:two;
  char hello:three;
} morning_t;

typedef double fii;

typedef fii foo[deux];

void generate06()
{
  int i = 0;
  double x = 1.;
  foo y;
  union either_or {
    int either;
    double or;
  } z;
  morning_t u;
  extern fii func();

  // use an undeclared function without source code, which returns
  // implictly an inta typedef type
  (void) func(y, z, u, &x);
}
