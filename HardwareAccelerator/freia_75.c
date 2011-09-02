#include <stdio.h>
#include "freia.h"

int freia_75(freia_data2d * out, const freia_data2d * in)
{
  int vol, pmin, pmax, x, y;
  freia_aipo_add_const(out, in, 1);
  freia_aipo_global_vol(out, &vol);
  freia_aipo_global_min(out, &pmin);
  freia_aipo_global_max_coord(out, &pmax, &x, &y);
  return vol/(pmin+pmax)*2;
}
