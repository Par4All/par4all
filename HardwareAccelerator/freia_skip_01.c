#include "freia.h"

freia_status
freia_skip_01(freia_data2d * o, freia_data2d * i)
{
  freia_status ret = FREIA_OK;
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  int32_t max, seuil;
  freia_aipo_global_max(i, &max);
  freia_aipo_sub_const(t, i, 20);
  seuil = max - 10;
  freia_aipo_threshold(o, t, 0, seuil, false);
  freia_common_destruct_data(t);
  return ret;
}
