#include "freia.h"

freia_status
freia_scalar_05(freia_data2d * o, freia_data2d * i, uint32_t * k)
{
  freia_data2d
    * X = freia_common_create_data(16, 128, 128),
    * tB = freia_common_create_data(16, 128, 128),
    * tC = freia_common_create_data(16, 128, 128),
    * tC2 = freia_common_create_data(16, 128, 128),
    * tD = freia_common_create_data(16, 128, 128);
  int sA, sB, sC;

  // look for order issues when reusing a temporary image
  freia_aipo_global_max(i, &sA);
  freia_aipo_threshold(tB, i, sA-10, sA-1, false);
  freia_aipo_erode_8c(X, i, k);
  freia_aipo_global_min(X, &sB);
  freia_aipo_threshold(tC, X, sB-10, sB-1, false);
  freia_aipo_mul(tC2, tC, tC);
  freia_aipo_dilate_8c(X, tB, k);
  freia_aipo_global_vol(X, &sC);
  freia_aipo_threshold(tD, X, sC-10, sC-1, false);
  freia_aipo_add(o, tC2, tD);

  freia_common_destruct_data(X);
  freia_common_destruct_data(tB);
  freia_common_destruct_data(tC);
  freia_common_destruct_data(tC2);
  freia_common_destruct_data(tD);
  return FREIA_OK;
}
