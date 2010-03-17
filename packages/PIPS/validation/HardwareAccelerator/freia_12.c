#include "freia.h"

freia_status
freia_12(freia_data2d * o0, freia_data2d * o1,
	 freia_data2d * i0, freia_data2d * i1,
	 int32_t kern)
{
  // two parallel computations
  // o0 = h(i0)
  // o1 = h(i1)
  freia_aipo_erode_8c(o0, i0, kern);
  freia_aipo_dilate_8c(o1, i1, kern);
  return FREIA_OK;
}
