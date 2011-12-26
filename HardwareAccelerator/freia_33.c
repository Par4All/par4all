#include "freia.h"

freia_status
freia_33(freia_data2d * o, freia_data2d * i, uint32_t * k)
{
  freia_data2d
    * X0 = freia_common_create_data(16, 128, 128),
    * X1 = freia_common_create_data(16, 128, 128),
    * tB = freia_common_create_data(16, 128, 128),
    * tB2 = freia_common_create_data(16, 128, 128),
    * tC = freia_common_create_data(16, 128, 128),
    * tC2 = freia_common_create_data(16, 128, 128),
    * tD = freia_common_create_data(16, 128, 128);
  int sA, sB, sC;

  // complex dag for split
  // A
  freia_aipo_global_max(i, &sA);
  // B
  freia_aipo_threshold(tB, i, sA-10, sA-1, false);
  // A
  freia_aipo_erode_8c(X0, i, k);
  // A
  freia_aipo_global_min(X0, &sB);
  // C
  freia_aipo_threshold(tC, X0, sB-10, sB-1, false);
  freia_aipo_mul(tC2, tC, tC);
  // B
  freia_aipo_dilate_8c(tB2, tB, k);
  freia_aipo_div(X1, tB2, tB);
  freia_aipo_global_vol(X1, &sC);
  // D
  freia_aipo_threshold(tD, X1, sC-10, sC-1, false);
  freia_aipo_add(o, tC2, tD);

  freia_common_destruct_data(X0);
  freia_common_destruct_data(X1);
  freia_common_destruct_data(tB);
  freia_common_destruct_data(tB2);
  freia_common_destruct_data(tC);
  freia_common_destruct_data(tC2);
  freia_common_destruct_data(tD);
  return FREIA_OK;
}
