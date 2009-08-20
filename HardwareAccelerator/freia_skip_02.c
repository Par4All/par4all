#include "freia.h"

void freia_skip_02(freia_data2d * o, freia_data2d * i)
{
  freia_data2d
    * t0 = freia_common_create_data(16, 128, 128),
    * t1 = freia_common_create_data(16, 128, 128);
  int32_t min, seuil;
  // div_const must be reordered *before* seuil assignment...
  freia_aipo_global_min(i, &min);
  freia_aipo_and_const(t0, i, 127);
  seuil = min - 10;
  freia_aipo_div_const(t1, t0, 3);
  freia_aipo_threshold(o, t1, seuil, 210, false);
  // cleanup
  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
}
