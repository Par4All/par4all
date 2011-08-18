#include "freia.h"

freia_status cst_02(freia_data2d * o, const freia_data2d * i)
{
  freia_data2d *t1, *t2;
  t1 = freia_common_create_data(i->bpp, i->widthWa, i->heightWa);
  t2 = freia_common_create_data(i->bpp, i->widthWa, i->heightWa);
  // o := 0
  freia_aipo_add_const(t1, i, 17);
  freia_aipo_sub_const(t2, i, -17);
  freia_aipo_sub(o, t1, t2);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  return FREIA_OK;
}
