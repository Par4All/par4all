#include "freia.h"

int freia_while_00(freia_data2d * out, const freia_data2d * in)
{
  int volprevious, volcurrent;
  freia_aipo_dilate_8c(out, in, freia_morpho_kernel_8c);
  freia_aipo_erode_8c(out, out, freia_morpho_kernel_8c);
  freia_aipo_global_vol(out, &volcurrent);
  volprevious = volcurrent;
  freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
  freia_aipo_inf(out, out, in);
  freia_aipo_global_vol(out, &volcurrent);
  while (volcurrent!=volprevious) {
    volprevious = volcurrent;
    freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
    freia_aipo_inf(out, out, in);
    freia_aipo_global_vol(out, &volcurrent);
  }
  return 0;
}
