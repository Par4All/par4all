#include <stdio.h>
#include "freia.h"

int freia_71(freia_data2d * out, freia_data2d * in)
{
  freia_data2d * t0, *t1, *t2;
  const  int32_t kernel1x3[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};

  t0 = freia_common_create_data(16, 1024, 720);
  t1 = freia_common_create_data(16, 1024, 720);
  t2 = freia_common_create_data(16, 1024, 720);

  freia_aipo_erode_8c(t0, in, kernel1x3);
  freia_aipo_add_const(t1, t0, 1);
  freia_aipo_not(t2, in);
  freia_aipo_and(out, t2, t1);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);

  return FREIA_OK;
}
