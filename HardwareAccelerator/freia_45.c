#include <stdio.h>
#include "freia.h"

freia_status freia_45(freia_data2d * in, freia_data2d * out)
{
  freia_data2d *t0, *t1;
  t0 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t1 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  freia_aipo_copy(t0, in);
  freia_aipo_copy(out, t0);
  freia_aipo_copy(t1, out);
  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  return FREIA_OK;
}
