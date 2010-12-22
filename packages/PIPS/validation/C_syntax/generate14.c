// Check the generation for boolean functions because of trouble with
// anr999 in Transformations-New

#include <stdbool.h>

void generate14()
{
  int i = 0;
  double x = 1.;
  bool b;
  extern bool func(int, double);

  b = func(i, x);
}
