#include <stdio.h>
#include "freia.h"

freia_status freia_48(freia_data2d *out, freia_data2d *in)
{
  freia_data2d
    *t0 = freia_common_create_data(16, 128, 128),
    *t1 = freia_common_create_data(16, 128, 128);

  freia_aipo_copy(t0, in);
  freia_aipo_mul_const(t0, t0, 3);
  freia_aipo_mul_const(out, out, 4);
  freia_aipo_add(out, out, t0);
  freia_aipo_div_const(t1, out, 7);
  freia_aipo_copy(out, t1);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  return FREIA_OK;
}
