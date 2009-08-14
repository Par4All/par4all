#include "freia.h"

freia_status
freia_14(freia_data2d * o, int32_t c)
{
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  // 2 alu to alu paths
  // t = h()
  // o = t + t
  freia_aipo_set_constant(t, c);
  freia_aipo_add(o, t, t);

  freia_common_destruct_data(t);
  return FREIA_OK;
}
