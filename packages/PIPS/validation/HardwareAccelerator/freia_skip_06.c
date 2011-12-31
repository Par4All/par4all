#include "freia.h"

void freia_skip_06(freia_data2d * o, freia_data2d * i)
{
  freia_data2d * t0, * t1;
  // live allocation in an intermediate statement
  freia_aipo_mul(i, i, i);
  t0 = freia_common_create_data(16, 128, 128);
  t1 = freia_common_create_data(16, 128, 128);
  freia_aipo_and_const(t0, i, 111);
  freia_aipo_xor_const(t1, i, 111);
  freia_aipo_addsat(o, t0, t1);
  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
}
