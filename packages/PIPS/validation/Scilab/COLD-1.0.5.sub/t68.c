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

  double _tmpxx0[1][10];
  for(int __tri0=0;__tri0 < 10;__tri0++) {
    _tmpxx0[0][__tri0] = 0.1+__tri0*0.1;
  }
  double _tmpxx1[1][10];
  
  for(int j=0; j<10; ++j) {
    _tmpxx1[0][j] = _tmpxx0[0][j];
  }
  double _u_x[10][1];
  scilab_rt_transposeConjugate_d2_d2(1,10,_tmpxx1,10,1,_u_x);
  double _u_y[10][1];
  scilab_rt_mul_d2i0_d2(10,1,_u_x,2,10,1,_u_y);
  scilab_rt_display_s0d2_("y",10,1,_u_y);
  double _u_z[10][1];
  scilab_rt_eltmul_d2d2_d2(10,1,_u_x,10,1,_u_x,10,1,_u_z);
  scilab_rt_display_s0d2_("z",10,1,_u_z);
  double _u_a[10][1];
  scilab_rt_div_d2i0_d2(10,1,_u_x,3,10,1,_u_a);
  scilab_rt_display_s0d2_("a",10,1,_u_a);
  double _tmpxx2[10][1];
  scilab_rt_add_d2d0_d2(10,1,_u_x,0.1,10,1,_tmpxx2);
  double _u_b[10][1];
  scilab_rt_eltdiv_d2d2_d2(10,1,_u_x,10,1,_tmpxx2,10,1,_u_b);
  scilab_rt_display_s0d2_("b",10,1,_u_b);
  double _u_c[10][1];
  scilab_rt_add_d2d2_d2(10,1,_u_x,10,1,_u_x,10,1,_u_c);
  scilab_rt_display_s0d2_("c",10,1,_u_c);

  scilab_rt_terminate();
}

