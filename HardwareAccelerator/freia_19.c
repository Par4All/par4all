#include "freia.h"

freia_status
freia_19(freia_data2d * o, freia_data2d * i)
{
  freia_data2d
    * t = freia_common_create_data(16, 128, 128);

  // one useless copy
  // t = i
  // o = t
  freia_aipo_copy(t, i);
  freia_aipo_copy(o, t);

  freia_common_destruct_data(t);

  return FREIA_OK;
}
