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

  /*  t134.sce: Testing function not  */
  double _u_a[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_a);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_a[1][j][k] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_a[i][1][k] = 20;
    }
  }

  for(int i=0; i<3;++i) {
    _u_a[i][1][2] = 30;
  }
  int _tmp0[3][2][3];
  scilab_rt_not_d3_i3(3,2,3,_u_a,3,2,3,_tmp0);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp0);
  double _u_b[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_b);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_b[1][j][k] = 10.1;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_b[i][1][k] = 20.5;
    }
  }

  for(int i=0; i<3;++i) {
    _u_b[i][1][2] = 30.6;
  }
  int _tmp1[3][2][3];
  scilab_rt_not_d3_i3(3,2,3,_u_b,3,2,3,_tmp1);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp1);
  int _u_c = (-10);
  int _tmp2 = (!_u_c);
  scilab_rt_display_s0i0_("ans",_tmp2);
  int _u_f[2][2];
  _u_f[0][0]=1;
  _u_f[0][1]=0;
  _u_f[1][0]=0;
  _u_f[1][1]=0;
  scilab_rt_display_s0i2_("f",2,2,_u_f);
  int _tmp3[2][2];
  scilab_rt_not_i2_i2(2,2,_u_f,2,2,_tmp3);
  scilab_rt_display_s0i2_("ans",2,2,_tmp3);

  scilab_rt_terminate();
}

