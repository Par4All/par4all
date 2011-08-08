#include "freia.h"

freia_status
freia_65(freia_data2d * o, const freia_data2d * i, int n)
{
  freia_data2d * t = freia_common_create_data(16, 128, 128);
  freia_aipo_copy(t, i); 
  for (int i=0; i<n; i++)
  {
    // loop carried dep on t
    freia_aipo_sub(o, o, t);
    freia_aipo_add_const(t, t, 1);
    freia_aipo_xor_const(o, o, 17);
  }
  freia_common_destruct_data(t);
  return FREIA_OK;
}
