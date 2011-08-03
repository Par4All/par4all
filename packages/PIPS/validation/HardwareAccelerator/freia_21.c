#include "freia.h"

freia_status
freia_21(freia_data2d * o, freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d * t = freia_common_create_data(i0->bpp, i0->widthWa, i0->heightWa);

  // internal variable reuses
  // t = i0 * i1
  // t = t - i0
  // o = t & i0
  freia_aipo_mul(t, i0, i1);
  freia_aipo_sub(t, t, i0);
  freia_aipo_and(o, t, i0);

  freia_common_destruct_data(t);

  return FREIA_OK;
}
