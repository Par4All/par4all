
#include <stdlib.h>
#include <stdio.h>

extern int scilab_rt_and_i2_(int si00, int si01, int in0[si00][si01])
{
  int lv0, lv1;
 
  int val = 0; 
  for (lv0=0; lv0<si00; lv0++) {
    for (lv1=0; lv1<si01; lv1++) {
      val += in0[lv0][lv1];
    }
  }
  return val;
}

