#include "freia.h"

void add_01(freia_data2d * o, const freia_data2d * i0, const freia_data2d * i1)
{
  freia_data2d * t;
  t = freia_common_create_data(16, 128, 128);
  freia_aipo_add(t, i0, i0);
  freia_aipo_add(o, i1, i1);
  freia_aipo_add(o, o, t);
  freia_common_destruct_data(t);
}
