#include "freia.h"

freia_status
freia_dup_10(freia_data2d * o, freia_data2d * t, freia_data2d * i0, freia_data2d * i1)
{
  // same operation performed twice, with commutation
  // t = inf(i0, i1)
  freia_aipo_inf(t, i0, i1);
  // o = inf(i1, i0)
  // could be replaced by o = t
  freia_aipo_inf(o, i1, i0);
  return FREIA_OK;
}
