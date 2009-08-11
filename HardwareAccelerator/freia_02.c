#include "freia.h"

freia_status
one_pipe(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
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

freia_status
two_pipes(freia_data2d * o0, freia_data2d * o1,
	  freia_data2d * i0, freia_data2d * i1)
{
  // two pipeline calls are necessary
  // o0 = i0 * i1
  // o1 = i0 & i1
  freia_aipo_mul(o0, i0, i1);
  freia_aipo_and(o1, i0, i1);
  return FREIA_OK;
}

freia_status
three_pipes(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d
    * t1 = freia_common_create_data(16, 128, 128),
    * t2 = freia_common_create_data(16, 128, 128);

  // 3 pipeline calls are necessary
  // t1 = i0 * i1
  // t2 = i0 & i1
  // o    = t1 | t2
  freia_aipo_mul(t1, i0, i1);
  freia_aipo_and(t2, i0, i1);
  freia_aipo_or(o, t1, t2);

  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);

  return FREIA_OK;
}
