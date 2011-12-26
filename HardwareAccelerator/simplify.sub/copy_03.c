#include "freia.h"

int copy_03(freia_data2d *out, const freia_data2d * in)
{
  freia_data2d * a, * b;
  a = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  b = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);

  // check for CSE-enabled constant images
  freia_aipo_add_const(a, in, 1);
  freia_aipo_add_const(b, in, 1);
  freia_aipo_sub(out, a, b);

  freia_common_destruct_data(a);
  freia_common_destruct_data(b);

  return 0;
}
