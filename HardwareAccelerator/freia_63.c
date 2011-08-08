#include "freia.h"

freia_status
freia_63(freia_data2d * o, const freia_data2d * i)
{
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  freia_aipo_cast(t, i); 
  freia_aipo_mul_const(o, t, 128);
  freia_common_destruct_data(t);
  return FREIA_OK;
}
