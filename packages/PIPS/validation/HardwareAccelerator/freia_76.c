#include <stdio.h>
#include "freia.h"

int freia_76(const freia_data2d * in)
{
  int vol, pmin, pmax, x, y;
  freia_aipo_global_max(in, &pmax);
  freia_aipo_global_vol(in, &vol);
  freia_aipo_global_min_coord(in, &pmin, &x, &y);
  return vol/(pmin+pmax)*2;
}
