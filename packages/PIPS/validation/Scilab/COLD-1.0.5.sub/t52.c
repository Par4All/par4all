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

  /*  t52.sce: testing abs function */
  int _tmpCT0 = scilab_rt_abs_i0_((-1));
  double _u_a = _tmpCT0;
  scilab_rt_display_s0d0_("a",_u_a);
  _u_a = scilab_rt_abs_d0_((-1.));
  scilab_rt_display_s0d0_("a",_u_a);
  double complex _tmpxx0 = (4*I);
  double complex _u_z1 = (3+_tmpxx0);
  double _u_b1 = scilab_rt_abs_z0_(_u_z1);
  scilab_rt_display_s0d0_("b1",_u_b1);
  double complex _tmpxx1 = (4*I);
  double complex _u_z2 = (3-_tmpxx1);
  double _u_b2 = scilab_rt_abs_z0_(_u_z2);
  scilab_rt_display_s0d0_("b2",_u_b2);
  int _tmpxx2 = (-3);
  double complex _tmpxx3 = (4*I);
  double complex _u_z3 = (_tmpxx2+_tmpxx3);
  double _u_b3 = scilab_rt_abs_z0_(_u_z3);
  scilab_rt_display_s0d0_("b3",_u_b3);
  int _tmpxx4 = (-3);
  double complex _tmpxx5 = (4*I);
  double complex _u_z4 = (_tmpxx4-_tmpxx5);
  double _u_b4 = scilab_rt_abs_z0_(_u_z4);
  scilab_rt_display_s0d0_("b4",_u_b4);
  int _tmpxx6[1][21];
  for(int __tri0=0;__tri0 < 21;__tri0++) {
    _tmpxx6[0][__tri0] = -10+__tri0*1;
  }
  int _u_a2[1][21];
  
  for(int j=0; j<21; ++j) {
    _u_a2[0][j] = _tmpxx6[0][j];
  }
  int _u_c2[1][21];
  scilab_rt_abs_i2_i2(1,21,_u_a2,1,21,_u_c2);
  scilab_rt_display_s0i2_("c2",1,21,_u_c2);
  double _u_a3[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_a3);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_a3[1][j][k] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_a3[i][1][k] = 20;
    }
  }

  for(int i=0; i<3;++i) {
    _u_a3[i][1][2] = 30;
  }
  double _tmpxx7[3][2][3];
  scilab_rt_sub_d3d0_d3(3,2,3,_u_a3,0.5,3,2,3,_tmpxx7);
  
  scilab_rt_assign_d3_d3(3,2,3,_tmpxx7,3,2,3,_u_a3);
  scilab_rt_display_s0d3_("a3",3,2,3,_u_a3);
  double _u_c3[3][2][3];
  scilab_rt_abs_d3_d3(3,2,3,_u_a3,3,2,3,_u_c3);
  scilab_rt_display_s0d3_("c3",3,2,3,_u_c3);

  scilab_rt_terminate();
}

