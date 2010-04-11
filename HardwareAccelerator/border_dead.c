#include "freia.h"

void border_dead(freia_data2d * out, freia_data2d * in)
{
  // this kernel is not very useful...
  const int32_t kernel = { 0, 0, 0,
			   0, 0, 0,
			   0, 0, 0 };
  freia_aipo_erode_8c(out, in, kernel);
}
