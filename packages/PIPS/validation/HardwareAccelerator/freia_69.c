#include "freia.h"

int freia_69(freia_data2d * o0, freia_data2d * o1, const freia_data2d * in)
{
  freia_data2d *t1, *t2;
  t1 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t2 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);

  freia_aipo_add_const(o0, in, 1);
  freia_aipo_add_const(o0, o0, 1);
  freia_aipo_add_const(o0, o0, 1);
  freia_aipo_add_const(o0, o0, 1);
  freia_aipo_add_const(o0, o0, 1);
  freia_aipo_add_const(o0, o0, 1);
  freia_aipo_add_const(o0, o0, 1);
  freia_aipo_erode_8c(o0, o0, freia_morpho_kernel_8c);
  freia_aipo_erode_8c(o0, o0, freia_morpho_kernel_8c);
  freia_aipo_sub(o0, in, o0);
  freia_aipo_not(t1, in);
  freia_aipo_inf_const(o1, t1, 255);
  freia_aipo_erode_8c(t2, t1, freia_morpho_kernel_8c);
  freia_aipo_inf(o1, o1, t2);

  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  return 0;
}
