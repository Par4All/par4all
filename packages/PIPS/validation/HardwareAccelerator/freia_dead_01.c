#include "freia.h"

freia_status
freia_dead_01(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d * tmp = freia_common_create_data(16, 128, 128);
  freia_status ret = FREIA_OK;
  ret |= freia_aipo_add(tmp, i0, i1);
  ret |= freia_aipo_add(o, i0, i1);
  ret |= freia_common_destruct_data(tmp);
  return FREIA_OK;
}
