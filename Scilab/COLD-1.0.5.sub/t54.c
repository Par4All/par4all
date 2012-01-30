/*
 * (c) HPC Project - 2010-2011 - All rights reserved
 *
 */

#include "scilab_rt.h"


int __lv0;
int __lv1;
int __lv2;
int __lv3;

/*----------------------------------------------------*/


/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  double _u_a[1][10];
  for(int __tri0=0;__tri0 < 10;__tri0++) {
    _u_a[0][__tri0] = 1+__tri0*1;
  }
  _u_a[0][(3-1)] = 30.;
  scilab_rt_display_s0d2_("a",1,10,_u_a);
  double _u_b[10][1];
  scilab_rt_transposeConjugate_d2_d2(1,10,_u_a,10,1,_u_b);
  _u_b[(4-1)][0] = 40;
  scilab_rt_display_s0d2_("b",10,1,_u_b);

  scilab_rt_terminate();
}

