#include "freia.h"

void freia_dup_13(freia_data2d * in0, int32_t * x, int32_t * y)
{
  // redudant measure
  freia_aipo_global_min(in0, x);
  freia_aipo_global_min(in0, y);
}
