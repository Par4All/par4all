#include "freia.h"

int copy_02(freia_data2d *o0, freia_data2d * o1, const freia_data2d * in)
{
  freia_data2d * a, * b, * c, * d;
  a = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  b = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  c = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  d = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);

  freia_aipo_copy(a, in);
  freia_aipo_copy(b, a);
  freia_aipo_copy(c, b);
  freia_aipo_add_const(d, c, 1);
  freia_aipo_copy(o0, d);
  freia_aipo_copy(o1, c);

  freia_common_destruct_data(a);
  freia_common_destruct_data(b);
  freia_common_destruct_data(c);
  freia_common_destruct_data(d);

  return 0;
}
