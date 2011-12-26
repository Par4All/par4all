#include "freia.h"

int copy_04(freia_data2d *out, const freia_data2d * in)
{
  freia_data2d * a, * b;
  a = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  b = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);

  // check for constant images which enable CSE
  freia_aipo_sub(a, in, in);
  freia_aipo_xor(b, in, in);

  // now these are common expressions
  freia_aipo_add(a, a, in);
  freia_aipo_add(b, b, in);

  // followed by some computations
  freia_aipo_add_const(a, a, 2);
  freia_aipo_add_const(b, b, 7);
  freia_aipo_mul(out, a, b);

  freia_common_destruct_data(a);
  freia_common_destruct_data(b);

  return 0;
}
