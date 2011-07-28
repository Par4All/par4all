#include "freia.h"

freia_status
freia_dead_02(void)
{
  freia_status ret = FREIA_OK;
  freia_data2d * tmp = freia_common_create_data(16, 128, 128);
  ret |= freia_common_destruct_data(tmp);
  return ret;
}
