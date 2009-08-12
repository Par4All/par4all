#include "freia.h"

// three pipes
freia_status
freia_06(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d
    * t1 = freia_common_create_data(16, 128, 128),
    * t2 = freia_common_create_data(16, 128, 128);

  // 3 pipeline calls are necessary
  // t1 = i0 * i1
  // t2 = i0 & i1
  // o  = t1 | t2
  freia_aipo_mul(t1, i0, i1);
  freia_aipo_and(t2, i0, i1);
  freia_aipo_or(o, t1, t2);

  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);

  return FREIA_OK;
}
