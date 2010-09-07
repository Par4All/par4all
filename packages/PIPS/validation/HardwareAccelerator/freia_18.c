#include "freia.h"

freia_status
freia_18(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d
    * u0 = freia_common_create_data(16, 128, 128),
    * u1 = freia_common_create_data(16, 128, 128),
    * u2 = freia_common_create_data(16, 128, 128),
    * u3 = freia_common_create_data(16, 128, 128),
    * t0 = freia_common_create_data(16, 128, 128);

  // useless computations...
  // u0 = i0 + i1
  // u1 = u0
  // u2 = u1 - i0
  // t0 = i0 & i1
  // o = t0 ^ i1
  // u3 = o | i1

  // deadcode
  freia_aipo_add(u0, i0, i1);
  freia_aipo_copy(u1, u0);
  freia_aipo_sub(u2, u1, i0);
  // alive code
  freia_aipo_and(t0, i0, i1);
  freia_aipo_xor(o, t0, i1);
  // deadcode
  freia_aipo_or(u3, o, i1);

  freia_common_destruct_data(u0);
  freia_common_destruct_data(u1);
  freia_common_destruct_data(u2);
  freia_common_destruct_data(u3);
  freia_common_destruct_data(t0);

  return FREIA_OK;
}
