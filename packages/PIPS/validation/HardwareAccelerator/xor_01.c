#include "freia.h"

freia_status xor_01(freia_data2d * o)
{
  // this means o := 7
  // freia_aipo_set_constant(o, 7);
  freia_aipo_xor(o, o, o);
  freia_aipo_add_const(o, o, 7);
  return FREIA_OK;
}
