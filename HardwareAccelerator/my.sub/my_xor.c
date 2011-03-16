#include "freia.h"
freia_status my_xor
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1)
{
  return freia_aipo_xor(o, i0, i1);
}
