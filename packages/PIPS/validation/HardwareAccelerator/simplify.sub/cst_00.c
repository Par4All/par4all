#include "freia.h"

freia_status cst_00(freia_data2d * o, uint32_t k)
{
  // this means o := 0
  freia_aipo_xor(o, o, o);
  freia_aipo_erode_8c(o, o, k);
  freia_aipo_copy(o, o);
  freia_aipo_dilate_6c(o, o, k);
  freia_aipo_add_const(o, o, 0);
  freia_aipo_sub_const(o, o, 0);
  freia_aipo_mul_const(o, o, 1);
  freia_aipo_div_const(o, o, 1);
  return FREIA_OK;
}
