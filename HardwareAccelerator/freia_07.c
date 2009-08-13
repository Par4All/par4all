#include "freia.h"

// AIPO reordering of 2 disconnected computations...
freia_status
freia_07(
  freia_data2d * o0, freia_data2d * o1,
  freia_data2d * i0, freia_data2d * i1,
  freia_data2d * i2, freia_data2d * i3)
{
  freia_data2d 
    * t0 = freia_common_create_data(16, 128, 128),
    * t1 = freia_common_create_data(16, 128, 128),
    * t2 = freia_common_create_data(16, 128, 128),
    * t3 = freia_common_create_data(16, 128, 128);

  // o0 = inf(i0 + i1, i0) - i0
  // o1 = sup(i2 / i3, i3) | i3
  freia_aipo_add(t0, i0, i1);
  freia_aipo_div(t2, i2, i3);
  freia_aipo_inf(t1, t0, i0);
  freia_aipo_sup(t3, t2, i3);
  freia_aipo_sub(o0, t1, i0);
  freia_aipo_or(o1, t3, i3);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  freia_common_destruct_data(t3);

  return FREIA_OK;
}
