#include "freia.h"

freia_status freia_55(freia_data2d * out, freia_data2d * in)
{
  freia_data2d * ta = freia_common_create_data(16, 128, 128);
  freia_data2d * tb = freia_common_create_data(16, 128, 128);
  freia_data2d * tc = freia_common_create_data(16, 128, 128);

  freia_aipo_set_constant(ta, 255);
  freia_aipo_dilate_8c(tb, in, freia_morpho_kernel_8c);
  freia_aipo_inf(tc, ta, tb);
  freia_aipo_erode_8c(tb, tb, freia_morpho_kernel_8c);
  freia_aipo_inf(out, tc, tb);

  freia_common_destruct_data(ta);
  freia_common_destruct_data(tb);
  freia_common_destruct_data(tc);
  return FREIA_OK;
}
