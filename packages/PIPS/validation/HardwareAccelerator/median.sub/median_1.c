#include "freia.h"

freia_status median_1(freia_data2d *o, freia_data2d *i)
{
  freia_status ret;
  freia_data2d * t = freia_common_create_data(i->bpp, i->widthWa, i->heightWa);

  // int32_t c = 8;
  // ret =  freia_cipo_close(t, i, c, 1);
  ret =  freia_aipo_dilate_8c(t, i, freia_morpho_kernel_8c);
  ret |= freia_aipo_erode_8c(t, t, freia_morpho_kernel_8c);

  // ret |= freia_cipo_open(t, t, c, 1);
  ret |= freia_aipo_erode_8c(t, t, freia_morpho_kernel_8c);
  ret |= freia_aipo_dilate_8c(t, t, freia_morpho_kernel_8c);

  // ret |= freia_cipo_close(t, t, c, 1);
  ret |= freia_aipo_dilate_8c(t, t, freia_morpho_kernel_8c);
  ret |= freia_aipo_erode_8c(t, t, freia_morpho_kernel_8c);

  ret |= freia_aipo_inf(o, t, i);

  // ret |= freia_cipo_open(t, i, c, 1);
  ret |= freia_aipo_erode_8c(t, t, freia_morpho_kernel_8c);
  ret |= freia_aipo_dilate_8c(t, t, freia_morpho_kernel_8c);

  // ret |= freia_cipo_close(t, t, c, 1);
  ret |= freia_aipo_dilate_8c(t, t, freia_morpho_kernel_8c);
  ret |= freia_aipo_erode_8c(t, t, freia_morpho_kernel_8c);

  // ret |= freia_cipo_open(t, t, c, 1);
  ret |= freia_aipo_erode_8c(t, t, freia_morpho_kernel_8c);
  ret |= freia_aipo_dilate_8c(t, t, freia_morpho_kernel_8c);

  ret |= freia_aipo_sup(o, o, t);

  // cleanup
  ret |= freia_common_destruct_data(t);
  return ret;
}
