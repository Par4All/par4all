#include "freia.h"

freia_status
freia_23(freia_data2d * o, freia_data2d * io, freia_data2d * i1)
{
  // external input variable reuse
  // o = io ^ i1
  // io' = o - io
  // o = o + io'
  freia_aipo_xor(o, io, i1);
  freia_aipo_sub(io, o, io);
  freia_aipo_add(o, o, io);

  return FREIA_OK;
}
