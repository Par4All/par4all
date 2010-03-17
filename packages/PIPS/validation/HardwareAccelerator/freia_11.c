#include "freia.h"

freia_status
freia_11(freia_data2d * o, freia_data2d * i)
{
  // input used twice
  // o = i - i
  freia_aipo_sub(o, i, i);
  return FREIA_OK;
}
