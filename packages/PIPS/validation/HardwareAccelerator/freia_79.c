#include <stdio.h>
#include "freia.h"

void freia_79(freia_data2d *imOut, const freia_data2d * imIn)
{
  freia_data2d *w1, *w2;
  w1 = freia_common_create_data(imIn->bpp, imIn->widthWa, imIn->heightWa);
  w2 = freia_common_create_data(imIn->bpp, imIn->widthWa, imIn->heightWa);

  freia_aipo_dilate_8c(w2, imIn, freia_morpho_kernel_8c);
  freia_aipo_sub(w1, imIn, w2);
  freia_aipo_mul_const(w2, w1, 2);
  freia_aipo_copy(imOut, w2);

  if (true)
    freia_aipo_add(imOut, imOut, w2);

  freia_common_destruct_data(w1);
  freia_common_destruct_data(w2);
}
