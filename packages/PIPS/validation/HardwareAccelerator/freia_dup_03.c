#include "freia.h"

freia_status
freia_dup_03(freia_data2d * o0, freia_data2d * o1, freia_data2d * i, int32_t c)
{
  // same operation performed twice
  // o0 = i +. c
  // o1 = i +. c
  freia_aipo_add_const(o0, i, c);
  freia_aipo_add_const(o1, i, c);
  return FREIA_OK;
}
