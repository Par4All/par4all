#include <stdio.h>
#include "freia.h"

int freia_58(freia_data2d * out, const freia_data2d * in)
{
  int ret, volprevious, volcurrent, i, j, k, l, m, n;
  ret = freia_aipo_global_vol(out, &volcurrent);
  do {
    volprevious = volcurrent;
    freia_aipo_dilate_8c(out, out, freia_morpho_kernel_8c);
    i = 1;
    k = 0;
    j = 0;
    j |= freia_aipo_inf(out, out, in);
    l = j;
    m |= l;
    m |= freia_aipo_global_vol(out, &volcurrent);
  }
  while (volcurrent!=volprevious);
  n = m;
  return n;
}
