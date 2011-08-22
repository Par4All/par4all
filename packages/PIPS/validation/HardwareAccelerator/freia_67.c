#include "freia.h"

void freia_67(
  freia_data2d * out,
  freia_data2d * tmp,
  const freia_data2d * in,
  const int32_t * kernel)
{
  freia_aipo_erode_8c(out, in, kernel);
  freia_aipo_erode_8c(tmp, in, kernel);
  freia_aipo_sub(tmp, in, tmp);
  freia_aipo_not(out, in);
}
