#include "freia.h"

freia_status freia_56(freia_data2d * out, freia_data2d * in)
{
  // same as freia_55, but out is used as a temporary
  freia_data2d * ta = freia_common_create_data(16, 128, 128);
  freia_data2d * tb = freia_common_create_data(16, 128, 128);

  freia_aipo_set_constant(ta, 255);
  freia_aipo_dilate_8c(out, in, freia_morpho_kernel_8c);
  freia_aipo_inf(tb, ta, out);
  freia_aipo_erode_8c(out, out, freia_morpho_kernel_8c);
  freia_aipo_inf(out, tb, out);

  freia_common_destruct_data(ta);
  freia_common_destruct_data(tb);
  return FREIA_OK;
}
