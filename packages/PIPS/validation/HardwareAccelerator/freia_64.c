#include "freia.h"

freia_status
freia_64(freia_data2d * o, const freia_data2d * i0, const freia_data2d * i1, bool b)
{
  // two statements that may be out of a sequence...
  if (b)
    freia_aipo_add(o, i0, i1);
  else
    freia_aipo_sub(o, i0, i1);
  return FREIA_OK;
}
