#include "freia.h"

freia_status freia_50(freia_data2d * out, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d * t1 = freia_common_create_data(16, 128, 128);
  freia_data2d * t2 = freia_common_create_data(16, 128, 128);

  freia_aipo_erode_8c(t1, i0, freia_morpho_kernel_8c);
  freia_aipo_inf(t2, i0, i1);
  freia_aipo_sub(out, t1, t2);

  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  return FREIA_OK;
}
