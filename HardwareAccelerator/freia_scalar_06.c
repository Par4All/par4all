#include "freia.h"

freia_status
freia_scalar_06(freia_data2d * o, freia_data2d * i, uint32_t * k)
{
  freia_data2d
    * X = freia_common_create_data(16, 128, 128),
    * tB = freia_common_create_data(16, 128, 128),
    * tC = freia_common_create_data(16, 128, 128),
    * tC2 = freia_common_create_data(16, 128, 128),
    * tC3 = freia_common_create_data(16, 128, 128),
    * tD = freia_common_create_data(16, 128, 128);
  int32_t sA, sA2, sB;

  // look for order issues when reusing a temporary image
  // A
  freia_aipo_global_max(i, &sA);
  // B
  freia_aipo_threshold(tB, i, sA-10, sA-1, false);
  // A
  freia_aipo_erode_8c(X, i, k);
  // A
  freia_aipo_global_min(X, &sA2);
  // C with X from A
  freia_aipo_threshold(tC, X, sA2-10, sA2-1, false);
  // C
  freia_aipo_dilate_6c(tC2, tC, k);
  // C
  freia_aipo_add(tC3, tC, tC2);
  // B
  freia_aipo_dilate_8c(X, tB, k);
  // B
  freia_aipo_global_vol(X, &sB);
  // D with X from B
  freia_aipo_threshold(tD, X, sB-10, sB-1, false);
  // D
  freia_aipo_add(o, tC3, tD);

  freia_common_destruct_data(X);
  freia_common_destruct_data(tB);
  freia_common_destruct_data(tC);
  freia_common_destruct_data(tD);
  return FREIA_OK;
}
