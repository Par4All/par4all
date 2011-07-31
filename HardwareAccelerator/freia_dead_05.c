#include <stdio.h>
#include "freia.h"

freia_data2d * freia_dead_05(void)
{
   freia_data2d * result = result = freia_common_create_data(16, 1024, 720);
   freia_aipo_set_constant(result, 127);
   return result;
}
