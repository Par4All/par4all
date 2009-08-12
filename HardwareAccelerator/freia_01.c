#include "freia.h"

freia_status
freia_01(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d 
    * tmp1 = freia_common_create_data(16, 128, 128),
    * tmp2 = freia_common_create_data(16, 128, 128);

  // o = inf(i0 + i1, i0) - i0
  freia_aipo_add(tmp1, i0, i1);
  freia_aipo_inf(tmp2, tmp1, i0);
  freia_aipo_sub(o, tmp2, i0);

  freia_common_destruct_data(tmp1);
  freia_common_destruct_data(tmp2);

  return FREIA_OK;
}
