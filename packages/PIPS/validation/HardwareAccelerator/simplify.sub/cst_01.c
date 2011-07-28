#include "freia.h"

freia_status cst_01(freia_data2d * o, uint32_t k, int32_t v)
{
  // this means o := 0
  freia_aipo_set_constant(o, v);
  freia_aipo_add(o, o, o);
  freia_aipo_dilate_6c(o, o, k);
  freia_aipo_add_const(o, o, v);
  freia_aipo_sub_const(o, o, v);
  freia_aipo_mul_const(o, o, v);
  freia_aipo_div_const(o, o, v);
  return FREIA_OK;
}
