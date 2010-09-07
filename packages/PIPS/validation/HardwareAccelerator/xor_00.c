#include "freia.h"

freia_status xor_00(freia_data2d * o)
{
  // this means o := 0
  // freia_aipo_set_constant(o, 0);
  freia_aipo_xor(o, o, o);
  return FREIA_OK;
}
