#include "freia.h"

freia_status
freia_27(freia_data2d * o0, freia_data2d * o1, freia_data2d * i, int32_t * k)
{
  // parallel reuses
  // o0 = i * i
  // o1 = f(i)
  freia_aipo_mul(o0, i, i);
  freia_aipo_erode_8c(o1, i, k);

  return FREIA_OK;
}
