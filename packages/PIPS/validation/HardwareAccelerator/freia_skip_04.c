#include "freia.h"

void freia_skip_04(freia_data2d * o, freia_data2d * i)
{
  int32_t max, seuil;
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  // use some control in an intermediate statement
  freia_aipo_global_max(i, &max);
  freia_aipo_xor_const(t, i, 111);
  if (max > 128)
    seuil = max - 10;
  else
    seuil = max + 15;
  freia_aipo_threshold(o, t, 8, seuil-4, true);
  freia_common_destruct_data(t);
}
