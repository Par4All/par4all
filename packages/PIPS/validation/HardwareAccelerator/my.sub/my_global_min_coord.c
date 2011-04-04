#include "freia.h"
freia_status my_global_min_coord
(freia_data2d * i0, int32_t * p0, int32_t * p1, int32_t * p2)
{
  return freia_aipo_global_min_coord(i0, p0, p1, p2);
}
