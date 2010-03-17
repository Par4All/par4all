#include "freia.h"

freia_status
freia_reuse_01(freia_data2d * o0, freia_data2d * o1, freia_data2d * i, int32_t * k)
{
  freia_data2d * t = freia_common_create_data(16, 128, 128);

  // variable reuse & duplicate operations?
  freia_aipo_erode_6c(o0, i, k);
  freia_aipo_erode_6c(t, i, k);
  freia_aipo_erode_6c(t, t, k);
  freia_aipo_dilate_6c(o1, i, k);
  freia_aipo_dilate_6c(o1, o1, k);
  freia_aipo_sub(o1, o1, t);

  //freia_common_destruct_data(t);

  return FREIA_OK;
}
