#include "freia.h"

// one multi-stage arithmetic pipe
freia_status
freia_03(freia_data2d * o,
  freia_data2d * i0, freia_data2d * i1,
  int32_t c1, int32_t c2, int32_t c3, int32_t c4, int32_t c5)
{
  freia_data2d
    * t1 = freia_common_create_data(16, 128, 128),
    * t2 = freia_common_create_data(16, 128, 128),
    * t3 = freia_common_create_data(16, 128, 128),
    * t4 = freia_common_create_data(16, 128, 128),
    * t5 = freia_common_create_data(16, 128, 128),
    * t6 = freia_common_create_data(16, 128, 128),
    * t7 = freia_common_create_data(16, 128, 128);

  // 1 pipeline call is enough
  // t1 = f(i0)
  // t2 = g(i1)
  // t3 = t1 + t2
  // t4 = h(t1)
  // t5 = i(t3)
  // t6 = t4 & t5
  // t7 = m()
  // o  = t6 * t7
  freia_aipo_add_const(t1, i0, c1);
  freia_aipo_addsat_const(t2, i1, c2);
  freia_aipo_add(t3, t1, t2);
  freia_aipo_mul_const(t4, t1, c3);
  freia_aipo_div_const(t5, t3, c4);
  freia_aipo_and(t6, t4, t5);
  freia_aipo_set_constant(t7, c5);
  freia_aipo_mul(o, t6, t7);

  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  freia_common_destruct_data(t3);
  freia_common_destruct_data(t4);
  freia_common_destruct_data(t5);
  freia_common_destruct_data(t6);
  freia_common_destruct_data(t7);

  return FREIA_OK;
}
