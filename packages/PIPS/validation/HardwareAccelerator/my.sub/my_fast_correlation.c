#include "freia.h"
freia_status my_fast_correlation
(freia_data2d *o, freia_data2d *i0, freia_data2d *i1, uint32_t p0)
{
  return freia_aipo_fast_correlation(o, i0, i1, p0);
}
