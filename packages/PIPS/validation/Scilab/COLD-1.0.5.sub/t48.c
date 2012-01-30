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

  int _tmpxx0[1][1000];
  for(int __tri0=0;__tri0 < 1000;__tri0++) {
    _tmpxx0[0][__tri0] = 1+__tri0*1;
  }
  int _u_a[1][1000];
  
  for(int j=0; j<1000; ++j) {
    _u_a[0][j] = _tmpxx0[0][j];
  }
  int _u_b;
  scilab_rt_max_i2_i0(1,1000,_u_a,&_u_b);
  scilab_rt_display_s0i0_("b",_u_b);
  /* c = [-1e60 -1e42 -1e80]; */
  double _u_c[1][3];
  _u_c[0][0]=(-1E15);
  _u_c[0][1]=(-1E12);
  _u_c[0][2]=(-1E30);
  double _u_d;
  scilab_rt_max_d2_d0(1,3,_u_c,&_u_d);
  scilab_rt_display_s0d0_("d",_u_d);

  scilab_rt_terminate();
}

