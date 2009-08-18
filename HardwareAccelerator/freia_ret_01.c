#include "freia.h"

freia_status
freia_ret_01(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_status ret1, ret2 = FREIA_OK;
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  ret1 = freia_aipo_addsat(t, i0, i1);
  ret2 |= freia_aipo_sub(o, i0, t);
  freia_common_destruct_data(t);
  return ret1 | ret2;
}
