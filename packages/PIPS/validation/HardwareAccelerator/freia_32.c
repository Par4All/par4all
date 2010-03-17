#include "freia.h"

freia_status
freia_32(freia_data2d * o0, freia_data2d * o1, freia_data2d * i0, freia_data2d * i1)
{
  // test 2 independent alu operations
  // o0 = i0 +. 10
  // o1 = i1 ^. 123
  freia_aipo_add_const(o0, i0, 10);
  freia_aipo_xor_const(o1, i1, 123);
  return FREIA_OK;
}
