#include "freia.h"

freia_status
  freia_scalar_04(freia_data2d * o,
		  freia_data2d * i0, freia_data2d * i1,
		  int32_t * k)
{
  freia_data2d
    * t1 = freia_common_create_data(16, 128, 128),
    * t2 = freia_common_create_data(16, 128, 128);
  int max, min = 10;
  bool bin = false;
  // break pipeline on scalar dependency
  freia_aipo_add(t1, i0, i1);
  freia_aipo_global_max(t1, &max);
  freia_aipo_erode_8c(t2, t1, k);
  freia_aipo_threshold(o, t2, min, max, bin);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  return FREIA_OK;
}
