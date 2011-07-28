#include "freia.h"

freia_status
freia_30(freia_data2d * o0, freia_data2d * o1, freia_data2d * i)
{
  // parallel reuses (similar to freia_27)
  // o0 = f(i)
  // o1 = i + i
  freia_aipo_threshold(o0, i, 10, 20, true);
  freia_aipo_add(o1, i, i);

  return FREIA_OK;
}
