#include <stdio.h>
#include "freia.h"

int freia_dead_08(freia_data2d * out0, freia_data2d * out1,
                  const freia_data2d * in)
{
  freia_data2d *tmp;
  tmp = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);

  freia_aipo_dilate_8c(tmp, in, freia_morpho_kernel_8c);
  freia_aipo_add(out1, tmp, in);
  freia_aipo_copy(out0, out1);

  freia_common_destruct_data(tmp);
  return 0;
}
