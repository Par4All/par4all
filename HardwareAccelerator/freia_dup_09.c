#include "freia.h"

void freia_dup_09(freia_data2d * in0, int32_t * x)
{
  // artificial WW dependencies, could be merged...
  freia_aipo_global_min(in0, x);
  freia_aipo_global_min(in0, x);
}
