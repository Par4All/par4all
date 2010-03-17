#include "freia.h"

freia_status
  freia_scalar_01(freia_data2d * o, freia_data2d * i)
{
  int thr = 12;
  // expressions and constants as parameters
  freia_aipo_threshold(o, i, thr-10, thr+10, false);
  return FREIA_OK;
}
