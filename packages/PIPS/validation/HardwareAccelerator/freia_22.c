#include "freia.h"

freia_status
freia_22(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  // external variable reuse
  // o = i0 / i1
  // o = o | i0
  // o = o +sat i0
  freia_aipo_div(o, i0, i1);
  freia_aipo_or(o, o, i0);
  freia_aipo_addsat(o, o, i0);

  return FREIA_OK;
}
