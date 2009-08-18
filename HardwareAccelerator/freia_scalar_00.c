#include "freia.h"

freia_status
  freia_scalar_00(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d * t1 = freia_common_create_data(16, 128, 128);
  int max, min = 10;
  bool bin = false;
  // break pipeline on scalar dependency
  freia_aipo_global_max(i0, &max);
  freia_aipo_add(t1, i0, i1);
  freia_aipo_threshold(o, t1, min, max, bin);
  freia_common_destruct_data(t1);
  return FREIA_OK;
}
