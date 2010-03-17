#include "freia.h"

freia_status
freia_29(freia_data2d * o0, freia_data2d * o1, freia_data2d * i, int32_t * k)
{
  // parallel reuses (similar to freia_27)
  // o0 = f(i)
  // o1 = i | i
  freia_aipo_erode_6c(o0, i, k);
  freia_aipo_or(o1, i, i);

  return FREIA_OK;
}
