#include <stdio.h>
#include "freia.h"

int freia_74(freia_data2d * out, const freia_data2d * in)
{
  freia_data2d *t0, *t1, *t2, *t3;
  const int32_t kernel1x3[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
  register freia_status ret_0, ret_1;

  t0 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t1 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t2 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  t3 = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);

  ret_0 = freia_aipo_erode_8c(t0, in, kernel1x3);
  ret_1 = ret_0;
  freia_aipo_dilate_8c(t1, in, kernel1x3);
  freia_aipo_add_const(t2, t0, 1);
  freia_aipo_mul_const(t3, t1, 2);
  freia_aipo_and(out, t2, t3);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  freia_common_destruct_data(t3);

  return 0;
}
