#include "freia.h"

int main(freia_data2d * img)
{
  int vol, bla;
  int * p;
  p = &bla;
  freia_aipo_global_vol(img, &vol);
  *p = vol;
}
