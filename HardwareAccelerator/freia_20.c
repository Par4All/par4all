#include "freia.h"

freia_status
freia_20(freia_data2d * o, freia_data2d * i, int32_t * k)
{
  freia_data2d
    * t0 = freia_common_create_data(16, 128, 128),
    * t1 = freia_common_create_data(16, 128, 128),
    * t2 = freia_common_create_data(16, 128, 128),
    * t3 = freia_common_create_data(16, 128, 128),
    * t4 = freia_common_create_data(16, 128, 128),
    * t5 = freia_common_create_data(16, 128, 128),
    * t6 = freia_common_create_data(16, 128, 128),
    * t7 = freia_common_create_data(16, 128, 128),
    * t8 = freia_common_create_data(16, 128, 128),
    * t9 = freia_common_create_data(16, 128, 128),
    * ta = freia_common_create_data(16, 128, 128),
    * tb = freia_common_create_data(16, 128, 128);

  // long pipeline that must be cut
  freia_aipo_erode_8c(t0, i, k);
  freia_aipo_erode_6c(t1, t0, k);
  freia_aipo_dilate_8c(t2, t1, k);
  freia_aipo_dilate_6c(t3, t2, k);
  freia_aipo_erode_8c(t4, t3, k);
  freia_aipo_erode_6c(t5, t4, k);
  freia_aipo_dilate_8c(t6, t5, k);
  freia_aipo_dilate_6c(t7, t6, k);
  // should cut here
  freia_aipo_erode_8c(t8, t7, k);
  freia_aipo_erode_6c(t9, t8, k);
  freia_aipo_dilate_8c(ta, t9, k);
  freia_aipo_dilate_6c(tb, ta, k);
  freia_aipo_erode_8c(o, tb, k);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  freia_common_destruct_data(t3);
  freia_common_destruct_data(t4);
  freia_common_destruct_data(t5);
  freia_common_destruct_data(t6);
  freia_common_destruct_data(t7);
  freia_common_destruct_data(t8);
  freia_common_destruct_data(t9);
  freia_common_destruct_data(ta);
  freia_common_destruct_data(tb);

  return FREIA_OK;
}
