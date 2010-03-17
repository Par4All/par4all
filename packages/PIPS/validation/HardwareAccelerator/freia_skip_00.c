#include "freia.h"

freia_status
freia_skip_00(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_status ret;
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  freia_aipo_subsat(t, i0, i1);
  ret = FREIA_OK;
  freia_aipo_xor(o, i0, t);
  ret |= FREIA_OK;
  freia_common_destruct_data(t);
  return ret;
}
