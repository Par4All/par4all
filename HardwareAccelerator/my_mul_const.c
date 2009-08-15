#include "freia.h"
freia_status my_mul_const
(freia_data2d *o, freia_data2d *i0, int32_t p0)
{
  return freia_aipo_mul_const(o, i0, p0);
}
