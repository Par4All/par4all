#include "freia.h"

freia_status freia_53(freia_data2d * out, freia_data2d * in)
{
  freia_data2d * tb = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  freia_data2d * tc = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  freia_data2d * td = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);

  freia_aipo_dilate_8c(tb, in, freia_morpho_kernel_8c);
  freia_aipo_inf_const(tc, tb, 255);
  freia_aipo_erode_8c(tb, tb, freia_morpho_kernel_8c);
  freia_aipo_inf(td, tc, tb);
  freia_aipo_sub(out, in, td);

  freia_common_destruct_data(tb);
  freia_common_destruct_data(tc);
  freia_common_destruct_data(td);
  return FREIA_OK;
}
