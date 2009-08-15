#include "freia.h"

freia_status
freia_16(freia_data2d * o0, freia_data2d * o1, freia_data2d * i,
         int32_t c, int32_t *k, int32_t * r0, int32_t *r1, int32_t *r2)
{
  // test re ordering of mes
  freia_aipo_add_const(o0, i, c);
  freia_aipo_erode_6c(o1, i, k);
  freia_aipo_global_min(i, r0);
  freia_aipo_global_max(i, r1);
  freia_aipo_global_vol(i, r2);
  return FREIA_OK;
}
