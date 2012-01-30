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

  /*  t55.sce: testing sign function */
  double _tmp0 = scilab_rt_sign_i0_(1);
  scilab_rt_display_s0d0_("ans",_tmp0);
  double _tmp1 = scilab_rt_sign_i0_((-1));
  scilab_rt_display_s0d0_("ans",_tmp1);
  double complex _tmp2 = scilab_rt_sign_z0_((1+I));
  scilab_rt_display_s0z0_("ans",_tmp2);
  double complex _tmp3 = scilab_rt_sign_z0_((1-I));
  scilab_rt_display_s0z0_("ans",_tmp3);
  double _tmpxx0[2][3];
  scilab_rt_ones_i0i0_d2(2,3,2,3,_tmpxx0);
  double _u_a2[2][3];
  scilab_rt_sub_d2d0_d2(2,3,_tmpxx0,0.3,2,3,_u_a2);
  scilab_rt_display_s0d2_("a2",2,3,_u_a2);

  for(int j=0; j<3;++j) {
    _u_a2[1][j] = 10;
  }

  for(int i=0; i<2;++i) {
    _u_a2[i][1] = 20;
  }
  double _tmpxx1[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_tmpxx1);
  double _u_a3[3][2][3];
  scilab_rt_sub_d3d0_d3(3,2,3,_tmpxx1,0.2,3,2,3,_u_a3);

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
  _u_a3[0][1][0] = 0;
  scilab_rt_display_s0d3_("a3",3,2,3,_u_a3);
  double _tmp4[2][3];
  scilab_rt_sign_d2_d2(2,3,_u_a2,2,3,_tmp4);
  scilab_rt_display_s0d2_("ans",2,3,_tmp4);
  double _tmp5[3][2][3];
  scilab_rt_sign_d3_d3(3,2,3,_u_a3,3,2,3,_tmp5);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp5);
  double _u_z[2][3];
  scilab_rt_zeros_i0i0_d2(2,3,2,3,_u_z);
  scilab_rt_display_s0d2_("z",2,3,_u_z);
  double _tmp6[2][3];
  scilab_rt_sign_d2_d2(2,3,_u_z,2,3,_tmp6);
  scilab_rt_display_s0d2_("ans",2,3,_tmp6);

  scilab_rt_terminate();
}

