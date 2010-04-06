#include "freia.h"
freia_status freia_39(freia_data2d *o, freia_data2d *i, int32_t * kern)
{
  freia_data2d
    * e = freia_common_create_data(i->bpp, i->widthWa, i->heightWa),
    * d = freia_common_create_data(i->bpp, i->widthWa, i->heightWa);

  // preschedule example for terapix tests

  freia_aipo_erode_8c(e, i, kern);
  freia_aipo_dilate_8c(d, i, kern);

  freia_aipo_erode_8c(e, e, kern);
  freia_aipo_dilate_8c(d, d, kern);

  freia_aipo_sub(o, d, e);

  freia_common_destruct_data(e);
  freia_common_destruct_data(d);

  return FREIA_OK;
}
