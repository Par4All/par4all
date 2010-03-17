#include "freia.h"

// one pipe for compaction
freia_status
freia_04(freia_data2d * o, freia_data2d * i,
	 int32_t * k,
	 int32_t inf, int32_t sup, bool bin,
	 int32_t * m, int32_t * v)
{
  freia_data2d
    * t0 = freia_common_create_data(16, 128, 128),
    * t1 = freia_common_create_data(16, 128, 128),
    * t2 = freia_common_create_data(16, 128, 128),
    * t3 = freia_common_create_data(16, 128, 128),
    * t4 = freia_common_create_data(16, 128, 128);

  // to test operator compaction
  // t0 = erode(i)
  // t1 = dilate(i)
  // t2 = t1 - t0
  // t3 = threshold(t2)
  // t4 = threshold(t0)
  // v  = vol(t3)
  // m  = min(t4)
  // o  = t4 + t3
  freia_aipo_erode_8c(t0, i, k);
  freia_aipo_dilate_6c(t1, i, k);
  freia_aipo_sub(t2, t1, t0);
  freia_aipo_threshold(t3, t2, inf, sup, bin);
  freia_aipo_threshold(t4, t0, inf, sup, bin);
  freia_aipo_global_vol(t3, v);
  freia_aipo_global_min(t4, m);
  freia_aipo_add(o, t3, t4);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  freia_common_destruct_data(t3);
  freia_common_destruct_data(t4);

  return FREIA_OK;
}
