#include "freia.h"

freia_status
freia_02(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d
    * t1 = freia_common_create_data(16, 128, 128),
    * t2 = freia_common_create_data(16, 128, 128);

  // 1 pipeline call is enough
  // t1 = i1 / i0
  // t2 = t1 + i0
  // o  = t2 ^ i0
  freia_aipo_div(t1, i1, i0);
  freia_aipo_add(t2, t1, i0);
  freia_aipo_xor(o, t2, i0);

  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);

  return FREIA_OK;
}
