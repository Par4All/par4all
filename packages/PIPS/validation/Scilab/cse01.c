#include <stdlib.h>
#include "sci2clib.h"

/* Some extract from Scilab2C 2.0 on KleiBottle example */

void scilab2c() {
   double step_u;
  double __temp2;
  double* __temp3 = NULL;
  int ____temp3Size[2];
   double* __temp4 = NULL;
  int ____temp4Size[2];
  double* U = NULL;
  int __USize[2];

  ____temp3Size[0]=d0absd0(d0floord0((((d0reald0(__temp2)-0)/d0reald0(step_u))+1)));
  ____temp3Size[1]=d0absd0(d0floord0((((d0reald0(__temp2)-0)/d0reald0(step_u))+1)));
  __temp3 = (double*) realloc(__temp3,(d0absd0(d0floord0((((d0reald0(__temp2)-0)/d0reald0(step_u))+1))))*(d0absd0(d0floord0((((d0reald0(__temp2)-0)/d0reald0(step_u))+1))))*sizeof(double));
  d2cosd2(U,  __USize,__temp3);
  ____temp4Size[0]=d0absd0(d0floord0((((d0reald0(__temp2)-0)/d0reald0(step_u))+1)));
  ____temp4Size[1]=d0absd0(d0floord0((((d0reald0(__temp2)-0)/d0reald0(step_u))+1)));
  __temp4 = (double*) realloc(__temp4,(d0absd0(d0floord0((((d0reald0(__temp2)-0)/d0reald0(step_u))+1))))*(d0absd0(d0floord0((((d0reald0(__temp2)-0)/d0reald0(step_u))+1))))*sizeof(double));
  //d2d0OpDotSlashd2(U,  __USize,2,__temp4);
}
