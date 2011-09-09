#include "freia.h"

freia_status
freia_dup_04(freia_data2d * o0, freia_data2d * o1, freia_data2d * i,
  int32_t bas, int32_t haut, bool b)
{
  // same operation performed twice
  // o0 = thr(i)
  // o1 = thr(i)
  freia_aipo_threshold(o0, i, bas, haut, b);
  freia_aipo_threshold(o1, i, bas, haut, b);
  return FREIA_OK;
}
