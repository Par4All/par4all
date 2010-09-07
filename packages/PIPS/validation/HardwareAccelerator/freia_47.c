#include <stdio.h>
#include "freia.h"

freia_status freia_47(freia_data2d * in, freia_data2d * o1, freia_data2d * o2)
{
  freia_data2d *t;
  t = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  // useless copy that requires backwards and back-forwards propagation
  freia_aipo_add(t, in, o1);
  freia_aipo_mul(o2, t, in);
  freia_aipo_copy(o1, t);
  freia_common_destruct_data(t);
  return FREIA_OK;
}
