#include "freia.h"

void freia_skip_07(freia_data2d * o, freia_data2d * i)
{
  freia_data2d * t0, * t1, * t2;
  // live allocation in an intermediate statement
  t0 = freia_common_create_data(16, 128, 128);
  t1 = freia_common_create_data(16, 128, 128);
  t2 = freia_common_create_data(16, 128, 128);
  freia_aipo_mul(t0, i, i);
  freia_aipo_and_const(t1, t0, 111);
  freia_aipo_xor_const(t2, t0, 111);
  freia_aipo_addsat(o, t1, t2);
  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
}
