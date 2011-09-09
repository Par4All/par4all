#include "freia.h"

freia_status
freia_dup_05(freia_data2d * o0, freia_data2d * o1, freia_data2d * i0, freia_data2d * i1)
{
  // the internal difference is performed twice, and the pipeline is broken
  // o0 = i0 - i1
  // o1 = |i0 - i1|
  freia_aipo_absdiff(o0, i0, i1);
  freia_aipo_sub(o1, i0, i1);
  return FREIA_OK;
}
