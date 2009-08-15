#include "freia.h"

freia_status
freia_17(freia_data2d * o, freia_data2d * i)
{
  freia_data2d
    * t0 = freia_common_create_data(16, 128, 128),
    * t1 = freia_common_create_data(16, 128, 128),
    * t2 = freia_common_create_data(16, 128, 128),
    * t3 = freia_common_create_data(16, 128, 128),
    * t4 = freia_common_create_data(16, 128, 128),
    * t5 = freia_common_create_data(16, 128, 128);

  // useless copies
  // t0 = i
  // t1 = t0
  // t2 = t1
  // o = t2
  // t3 = t1
  // t4 = t0
  // t5 = o
  freia_aipo_copy(t0, i);
  freia_aipo_copy(t1, t0);
  freia_aipo_copy(t2, t1);
  freia_aipo_copy(o, t2);
  freia_aipo_copy(t3, t1);
  freia_aipo_copy(t4, t0);
  freia_aipo_copy(t5, o);

  freia_common_destruct_data(t0);
  freia_common_destruct_data(t1);
  freia_common_destruct_data(t2);
  freia_common_destruct_data(t3);
  freia_common_destruct_data(t4);
  freia_common_destruct_data(t5);

  return FREIA_OK;
}
