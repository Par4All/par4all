#include <stdio.h>
#include "freia.h"

void freia_77
(freia_data2d * out,
 const freia_data2d * i0,
 const freia_data2d * i1)
{
  freia_data2d *t0, *t1, *t2, *t3, *t4;
  t0 = freia_common_create_data(i0->bpp, i0->widthWa, i0->heightWa);
  t1 = freia_common_create_data(i0->bpp, i0->widthWa, i0->heightWa);
  t2 = freia_common_create_data(i0->bpp, i0->widthWa, i0->heightWa);
  t3 = freia_common_create_data(i0->bpp, i0->widthWa, i0->heightWa);

  freia_aipo_add_const(t0, i0, 2);
  freia_aipo_erode_8c(t1, t0, freia_morpho_kernel_8c);
  freia_aipo_dilate_8c(t2, i0, freia_morpho_kernel_8c);
  freia_aipo_sub(t3, t2, i1);
  freia_aipo_add(out, t3, t1);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  freia_common_destruct_data(t3);
}
