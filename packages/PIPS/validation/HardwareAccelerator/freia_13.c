#include "freia.h"

freia_status
freia_13(freia_data2d * o0, freia_data2d * o1,
	 int32_t * kern, int32_t c)
{
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  // two parallel computations without inputs
  // t = h()
  // o0 = f(t)
  // o1 = g(t)
  freia_aipo_set_constant(t, c);
  freia_aipo_erode_8c(o0, t, kern);
  freia_aipo_dilate_8c(o1, t, kern);

  freia_common_destruct_data(t);
  return FREIA_OK;
}
