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
one_big_pipe(freia_data2d * o,
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

freia_status
one_more_pipe(freia_data2d * o, freia_data2d * i,
	      int32_t * k,
	      int32_t inf, int32_t sup, bool bin,
	      int32_t * m, int32_t * v)
{
  freia_data2d *
    t0 = freia_common_create_data(16, 128, 128),
    t1 = freia_common_create_data(16, 128, 128),
    t2 = freia_common_create_data(16, 128, 128),
    t3 = freia_common_create_data(16, 128, 128),
    t4 = freia_common_create_data(16, 128, 128);

  // to test operator compaction
  // t0 = erode(i)
  // t1 = dilate(i)
  // t2 = t0 - t1
  // t3 = threshold(t2)
  // t4 = threshold(t0)
  // v  = vol(t3)
  // m  = min(t4)
  // o  = t4 + t3
  freia_aipo_erode_8c(t0, i, k);
  freia_aipo_dilate_6c(t1, i, k);
  freia_aipo_sub(t2, t1, t0);
  freia_aipo_threshold(t3, t2, inf, sup, bin);
  freia_aipo_threshold(t4, t0, inf, sup, bin);
  freia_aipo_global_vol(t3, v);
  freia_aipo_global_min(t4, m);
  freia_aipo_add(o, t3, t4);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  freia_common_destruct_data(t3);
  freia_common_destruct_data(t4);

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
  // o  = t1 | t2
  freia_aipo_mul(t1, i0, i1);
  freia_aipo_and(t2, i0, i1);
  freia_aipo_or(o, t1, t2);

  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);

  return FREIA_OK;
}
