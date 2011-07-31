#include <stdio.h>
#include "freia.h"

freia_status freia_dead_06
(freia_data2d * out, const freia_data2d * in, int32_t * k)
{
  freia_aipo_erode_8c(out, in, k);
  freia_aipo_dilate_8c(out, out, k);
  freia_aipo_not(out, in);
  return FREIA_OK;
}
