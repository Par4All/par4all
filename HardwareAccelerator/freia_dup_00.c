#include "freia.h"

int freia_dup_00(freia_data2d * i)
{
  int max1, max2;
  // same measure performed twice
  freia_aipo_global_max(i, &max1);
  freia_aipo_global_max(i, &max2);
  return max1+max2;
}
