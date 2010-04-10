#include "freia.h"

int copy_00(freia_data2d *o0, freia_data2d * o1, freia_data2d * in)
{
   freia_data2d * a, * b, * c, * d;
   a = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
   b = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
   c = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
   d = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);

   // 1
   freia_aipo_dilate_8c(a, in, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(c, in, freia_morpho_kernel_8c);

   // 2
   freia_aipo_dilate_8c(d, c, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(c, c, freia_morpho_kernel_8c);
   freia_aipo_dilate_8c(b, a, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(a, a, freia_morpho_kernel_8c);

   // 3
   freia_aipo_sub(o0, b, a);
   freia_aipo_dilate_8c(d, d, freia_morpho_kernel_8c);
   freia_aipo_erode_8c(c, c, freia_morpho_kernel_8c);

   // 4
   freia_aipo_sub(o1, d, c);

   return 0;
}
