#include "freia.h"

void freia_73(freia_data2d * o0, freia_data2d * o1, freia_data2d * o2, const freia_data2d * in0)
{
  freia_aipo_copy(o0, in0);
  freia_aipo_copy(o1, in0);
  freia_aipo_copy(o2, o0);
}
