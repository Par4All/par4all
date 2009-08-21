#include "freia.h"

freia_status freia_34(freia_data2d * o, freia_data2d * i, int32_t * k)
{
  freia_data2d
    * t1 = freia_common_create_data(16, 128, 128),
    * t2 = freia_common_create_data(16, 128, 128);

  // 1 stage pipeline
  // t1 = f(i)
  // t2 = g(i)
  // o  = t1 - t2
  freia_aipo_erode_8c(t1, i, k);
  freia_aipo_dilate_8c(t2, i, k);
  freia_aipo_sub(o, t1, t2);

  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);

  return FREIA_OK;
}
