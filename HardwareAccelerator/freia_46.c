#include <stdio.h>
#include "freia.h"

freia_status freia_46(freia_data2d * in, freia_data2d * out)
{
  freia_data2d *t;
  t = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  // useless copy that requires backwards propagation
  freia_aipo_add(t, in, out);
  freia_aipo_copy(out, t);
  freia_common_destruct_data(t);
  return FREIA_OK;
}
