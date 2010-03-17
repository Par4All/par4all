#include "freia.h"

freia_status
freia_36(freia_data2d * i0, freia_data2d * i1)
{
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  int32_t min;
  // should t be extracted?
  freia_aipo_add(t, i0, i1);
  freia_aipo_global_min(t, &min);
  freia_common_destruct_data(t);
  return FREIA_OK;
}
