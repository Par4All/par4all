#include "freia.h"

// output, input+output, input arguments
freia_status freia_57(freia_data2d * o, freia_data2d * io, freia_data2d * i)
{
  freia_aipo_add(o, io, i);
  freia_aipo_inf(io, o, io);
  return FREIA_OK;
}
