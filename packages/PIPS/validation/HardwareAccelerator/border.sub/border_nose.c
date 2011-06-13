#include "freia.h"

void border_nose(freia_data2d * out, freia_data2d * in)
{
  const int32_t kernel = { 1, 1, 1,
			   1, 1, 1,
			   1, 1, 1 };
  freia_aipo_erode_8c(out, in, kernel);
}
