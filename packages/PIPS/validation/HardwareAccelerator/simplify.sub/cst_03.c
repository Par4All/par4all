#include "freia.h"

freia_status cst_03(freia_data2d * o)
{
  freia_aipo_xor(o, o, o);
  freia_aipo_addsat_const(o, o, 1);
  freia_aipo_absdiff_const(o, o, 2);
  freia_aipo_subsat_const(o, o, 3);
  return FREIA_OK;
}
