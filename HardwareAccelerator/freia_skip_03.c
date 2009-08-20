#include "freia.h"

void freia_skip_03(freia_data2d * o, freia_data2d * i)
{
  freia_data2d
    * t0 = freia_common_create_data(16, 128, 128),
    * t1 = freia_common_create_data(16, 128, 128);
  int32_t vol, mvol, seuil;
  // sub_const must be reordered *before* seuil assignment...
  freia_aipo_global_vol(i, &vol);
  freia_aipo_sub_const(t0, i, 127);
  mvol = vol / 1000 + 1;
  freia_aipo_add_const(t1, t0, 3);
  seuil = mvol + 3;
  freia_aipo_threshold(o, t1, seuil, 250, false);
  // cleanup
  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
}
