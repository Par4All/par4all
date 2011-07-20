#include "freia.h"
freia_status my_replace_const
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1, const int32_t c)
{
  return freia_aipo_replace_const(o, i0, i1, c);
}
