#include "freia.h"

// i->widthWa, i->heightWa
#define SIZE (128)
// bit per pixel i->pbb
#define BPP   (16)

freia_status
freia_09(freia_data2d * o,
	 freia_data2d * i0,
	 freia_data2d * i1,
	 freia_data2d * i2)
{
  freia_data2d
    * g0 = freia_common_create_data(BPP, SIZE, SIZE),
    * g1 = freia_common_create_data(BPP, SIZE, SIZE),
    * g2 = freia_common_create_data(BPP, SIZE, SIZE),
    * d0 = freia_common_create_data(BPP, SIZE, SIZE),
    * d1 = freia_common_create_data(BPP, SIZE, SIZE);
  int connexity = 8;
  int size = 1;

  // T-rex motion detection
  // i0, i1, i2 are successive input images
  // the could be a loop with a pipeline
  freia_cipo_inner_gradient(g0, i0, connexity, size);
  freia_cipo_inner_gradient(g1, i1, connexity, size);
  freia_cipo_inner_gradient(g2, i2, connexity, size);
  freia_aipo_absdiff(d1, g2, g1);
  freia_aipo_absdiff(d0, g1, g0);
  freia_aipo_inf(o, d1, d0);

  freia_common_destruct_data(g0);
  freia_common_destruct_data(g1);
  freia_common_destruct_data(g2);
  freia_common_destruct_data(d0);
  freia_common_destruct_data(d1);

  return FREIA_OK;
}
