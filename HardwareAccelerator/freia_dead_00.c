#include "freia.h"

freia_status
freia_dead_00(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d * tmp = freia_common_create_data(16, 128, 128);
  freia_aipo_add(tmp, i0, i1);
  int32_t v = freia_common_get(tmp, 12, 12);
  freia_aipo_sub_const(o, i0, v);
  freia_common_destruct_data(tmp);
  return FREIA_OK;
}
