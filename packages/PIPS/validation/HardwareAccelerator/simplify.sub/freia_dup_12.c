#include "freia.h"

freia_status
freia_dup_12(freia_data2d * o, freia_data2d * t, freia_data2d * i0, freia_data2d * i1)
{
  // same operation performed twice
  // t = i0 + i1
  freia_aipo_add(t, i0, i1);
  // o = i0 + i1
  // could be replaced by o = t
  freia_aipo_add(o, i0, i1);
  return FREIA_OK;
}
