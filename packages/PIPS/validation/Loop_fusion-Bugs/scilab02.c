#include <math.h>

void f() {
  int __USize[2] = { 100, 101 };
  double U[__USize[0]*__USize[1]];
  double _return391;
  double __temp3[__USize[0]*__USize[1]];
  double __temp4[__USize[0]*__USize[1]];

  int i = 0;
  //PIPS generated variable
  int I_30;
  I_30 = __USize[0]*__USize[1];
  {
    {
      for(i = 0; i <= I_30-1; i += 1) {
	{
	  //PIPS generated variable
	  double F_212;
	  F_212 = U[i];
	  _return391 = cos(F_212);
	}
	__temp3[i] = _return391;
      }
    }
  }
  {
    for(i = 0; i <= I_30-1; i += 1)
      __temp4[i] = U[i]/2;
  }
}
