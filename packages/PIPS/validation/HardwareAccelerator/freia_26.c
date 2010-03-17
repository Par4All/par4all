#include "freia.h"

freia_status
freia_26(freia_data2d * o0, freia_data2d * o1, freia_data2d * i0, freia_data2d * i1)
{
  // parallel reuses
  // o0 = i0 - i0
  // o1 = i1 / i1
  freia_aipo_sub(o0, i0, i0);
  freia_aipo_div(o1, i1, i1);

  return FREIA_OK;
}
