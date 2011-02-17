#include "freia.h"
void freia_3muls(freia_data2d * o1, freia_data2d * o2, freia_data2d * in0, freia_data2d * in1)
{
  freia_data2d * t0 = freia_common_create_data(16,128,128);
  freia_aipo_mul(t0, in0, in1);
  freia_aipo_mul(o1, in0, in1);
  freia_aipo_mul(o2, t0, o1);
  freia_common_destruct_data(t0);
}
