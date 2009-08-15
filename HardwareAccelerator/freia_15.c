#include "freia.h"

freia_status
freia_15(freia_data2d * o, freia_data2d * i,
         int32_t * r0, int32_t *r1, int32_t *r2, int32_t c)
{
  // test re ordering of mes
  freia_aipo_add_const(o, i, c);
  freia_aipo_global_min(i, r0);
  freia_aipo_global_max(i, r1);
  freia_aipo_global_vol(i, r2);
  return FREIA_OK;
}
