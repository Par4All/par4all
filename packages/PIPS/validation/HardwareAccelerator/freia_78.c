#include "freia.h"
void freia_78(freia_data2d * out, const freia_data2d * in)
{
  freia_data2d *
    t = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  freia_aipo_copy(t, in);
  if (true)
    freia_aipo_add(t, in, in);
  freia_aipo_add_const(out, t, 2);
  freia_common_destruct_data(t);
}
