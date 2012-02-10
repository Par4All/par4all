#include "freia.h"

int freia_scalar_10(freia_data2d * o, const freia_data2d * i)
{
  int max0, max;
  freia_aipo_global_max(i, &max);
  max0 = max;
  freia_aipo_add_const(o, i, 1);
  freia_aipo_global_max(o, &max);
  return max+max0;
}
