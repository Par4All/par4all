#include "freia.h"

void add_00(freia_data2d * o, const freia_data2d * i)
{
  freia_aipo_add(o, i, i);
}
