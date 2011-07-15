#include "freia.h"
freia_status my_convolution_3(freia_data2d *o, freia_data2d *i, int32_t *k)
{
  return freia_aipo_convolution(o, i, k, 3, 3);
}
