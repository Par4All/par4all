#include "freia.h"

freia_status
freia_61(freia_data2d * o, const freia_data2d * i0, const freia_data2d * i1)
{
  freia_data2d * t1 = freia_common_create_data(16, 128, 128);
  freia_aipo_add(t1, i0, i1);
  freia_aipo_threshold(o, t1, 100, 200, 0);
  if (1)
    freia_aipo_sub(o, t1, o);

  freia_common_destruct_data(t1);
  return FREIA_OK;
}
