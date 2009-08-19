#include "freia.h"

freia_status
freia_31(freia_data2d * o0, freia_data2d * o1, freia_data2d * i)
{
  // parallel reuses (similar to freia_27)
  // o0 = i ^ i
  // o1 = f(i)
  freia_aipo_xor(o0, i, i);
  freia_aipo_threshold(o1, i, 10, 20, true);

  return FREIA_OK;
}
