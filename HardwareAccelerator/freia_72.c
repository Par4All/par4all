#include <stdio.h>
#include "freia.h"

freia_status freia_72(freia_data2d * out, const freia_data2d * in)
{
  freia_data2d *t0, *t1, *t2, *t3, *t4;
  const  int32_t kernel1x3[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};

  t0 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t1 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t2 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t3 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t4 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);

  freia_aipo_erode_8c(t0, in, kernel1x3);
  freia_aipo_copy(t4, t0);

  freia_common_destruct_data(t4);

  freia_aipo_dilate_8c(t1, in, kernel1x3);
  freia_aipo_add_const(t2, t0, 1);
  freia_aipo_not(t3, t1);
  freia_aipo_and(out, t2, t3);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  freia_common_destruct_data(t3);

  return FREIA_OK;
}
