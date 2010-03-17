#include "freia.h"

freia_status
freia_35(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  int32_t min;
  freia_aipo_add(t, i0, i1);
  freia_aipo_sub(o, t, i1);
  freia_aipo_global_min(o, &min);
  freia_common_destruct_data(t);
  return FREIA_OK;
}
