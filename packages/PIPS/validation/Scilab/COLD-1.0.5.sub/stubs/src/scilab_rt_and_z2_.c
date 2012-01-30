
#include <stdlib.h>
#include <stdio.h>

#include <complex.h>

extern int scilab_rt_and_z2_(int si00, int si01, double complex in0[si00][si01])
{
  int lv0, lv1;
 
  double valR = 0; 
  double valI = 0; 
  for (lv0=0; lv0<si00; lv0++) {
    for (lv1=0; lv1<si01; lv1++) {
      valR += creal(in0[lv0][lv1]);
      valI += cimag(in0[lv0][lv1]);
    }
  }
  return (int) (valR + valI);
}

