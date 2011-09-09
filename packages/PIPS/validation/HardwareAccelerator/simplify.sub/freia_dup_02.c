#include "freia.h"

freia_status
freia_dup_02(freia_data2d * o0, freia_data2d * o1, freia_data2d * i, int32_t * k)
{
  // same operation performed twice
  // o0 = D6(i)
  // o1 = D6(i)
  freia_aipo_dilate_6c(o0, i, k);
  freia_aipo_dilate_6c(o1, i, k);
  return FREIA_OK;
}
