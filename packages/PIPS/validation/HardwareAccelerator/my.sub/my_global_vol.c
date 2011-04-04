#include "freia.h"
freia_status my_global_vol(freia_data2d *image, int32_t *vol)
{
  return freia_aipo_global_vol(image, vol);
}
