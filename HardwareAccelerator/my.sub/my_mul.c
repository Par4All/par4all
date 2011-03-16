#include "freia.h"
freia_status my_mul
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_mul(o, i0, i1);
}
