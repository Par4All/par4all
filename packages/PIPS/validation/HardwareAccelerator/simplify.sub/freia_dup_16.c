#include "freia.h"

int freia_dup_16(freia_data2d * i)
{
  int max1, max2, x, y;
  // same measure performed twice
  freia_aipo_global_max_coord(i, &max2, &x, &y);
  freia_aipo_global_max(i, &max1);
  return max1+max2+x+y;
}
