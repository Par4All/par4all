#include "freia.h"

freia_status
  freia_scalar_02(freia_data2d * o, freia_data2d * i0, freia_data2d * i1, int32_t * pmax)
{
  freia_data2d * tmp = freia_common_create_data(16, 128, 128);
  // dependency with a dereferencement
  freia_aipo_add(tmp, i0, i1);
  freia_aipo_global_max(tmp, pmax);
  freia_aipo_threshold(o, tmp, 10, *pmax, false);
  freia_common_destruct_data(tmp);
  return FREIA_OK;
}
