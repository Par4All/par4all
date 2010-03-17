#include "freia.h"

freia_status
freia_reuse_00(freia_data2d * o, freia_data2d * i, int32_t * k)
{
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  // reuse same variable over and over
  // t = E6(i)
  // t = D6(t)
  // t = E8(t)
  // o = D8(t)
  freia_aipo_erode_6c(t, i, k);
  freia_aipo_dilate_6c(t, t, k);
  freia_aipo_erode_8c(t, t, k);
  freia_aipo_dilate_8c(o, t, k);

  freia_common_destruct_data(t);

  return FREIA_OK;
}
