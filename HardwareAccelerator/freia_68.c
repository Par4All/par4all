#include "freia.h"

void freia_68(
  freia_data2d * o0,
  freia_data2d * o1,
  const freia_data2d * in)
{
  freia_data2d * tmp = freia_common_create_data(16, 128, 128);
  freia_aipo_add_const(tmp, in, 1);
  freia_aipo_sub(o0, in, tmp);
  freia_aipo_not(o1, in);
  freia_common_destruct_data(tmp);
}
