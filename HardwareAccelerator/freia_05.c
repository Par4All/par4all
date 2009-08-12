#include "freia.h"

// two pipe calls
freia_status
freia_05(freia_data2d * o0, freia_data2d * o1,
	 freia_data2d * i0, freia_data2d * i1)
{
  // two pipeline calls are necessary
  // o0 = i0 * i1
  // o1 = i0 & i1
  freia_aipo_mul(o0, i0, i1);
  freia_aipo_and(o1, i0, i1);
  return FREIA_OK;
}
