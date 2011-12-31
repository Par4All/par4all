#include "freia.h"

freia_status freia_25(freia_data2d * o0, freia_data2d * o1, const freia_data2d * i)
{
  // parallel reuses
  // o0 = i + i
  // o1 = i * i
  freia_aipo_add(o0, i, i);
  freia_aipo_mul(o1, i, i);

  return FREIA_OK;
}
