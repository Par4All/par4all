#include "freia.h"

void border_e(freia_data2d * out, freia_data2d * in)
{
  const int32_t kernel = { 0, 0, 0,
			   0, 1, 1,
			   0, 0, 0 };
  freia_aipo_erode_8c(out, in, kernel);
}
