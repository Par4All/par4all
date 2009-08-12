#include "freia.h"

freia_status
freia_00(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d * tmp = freia_common_create_data(16, 128, 128);
  freia_aipo_add(tmp, i0, i1);
  freia_aipo_sub(o, tmp, i1);
  freia_common_destruct_data(tmp);
  return FREIA_OK;
}
