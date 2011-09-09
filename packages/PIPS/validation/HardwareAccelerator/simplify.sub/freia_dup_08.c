#include "freia.h"

int freia_dup_08(freia_data2d * i)
{
  int max1, max2, x, y;
  // same measure performed twice
  freia_aipo_global_max(i, &max1);
  freia_aipo_global_max_coord(i, &max2, &x, &y);
  return max1+max2+x+y;
}
