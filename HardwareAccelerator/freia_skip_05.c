#include "freia.h"

void freia_skip_05(freia_data2d * o, freia_data2d * i)
{
  int32_t seuil;
  freia_data2d * t;
  // allocation in the middle of the stream
  freia_aipo_sub(i, i, i);
  t = freia_common_create_data(16, 128, 128);
  freia_aipo_xor_const(t, i, 111);
  freia_aipo_threshold(o, t, 8, 100, true);
  freia_common_destruct_data(t);
}
