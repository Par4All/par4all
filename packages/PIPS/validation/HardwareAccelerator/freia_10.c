#include "freia.h"

freia_status
freia_10(freia_data2d * o, freia_data2d * i, int32_t * k)
{
  freia_data2d
    * t0 = freia_common_create_data(16, 128, 128),
    * t1 = freia_common_create_data(16, 128, 128);

  // input used twice
  // o = erode(i) - dilate(i)
  freia_aipo_erode_8c(t0, i, k);
  freia_aipo_dilate_8c(t1, i, k);
  freia_aipo_sub(o, t1, t0);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);

  return FREIA_OK;
}
