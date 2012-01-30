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

  /*  t128.sce: Testing ne for 3D matrices */
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
  double _u_b[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_b);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_b[1][j][k] = 20;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_b[i][1][k] = 30;
    }
  }

  for(int i=0; i<3;++i) {
    _u_b[i][1][2] = 10;
  }
  double _u_c[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_c);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_c[1][j][k] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_c[i][1][k] = 30;
    }
  }

  for(int i=0; i<3;++i) {
    _u_c[i][1][2] = 20;
  }
  double _u_d[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_d);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_d[1][j][k] = 30;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_d[i][1][k] = 20;
    }
  }

  for(int i=0; i<3;++i) {
    _u_d[i][1][2] = 20;
  }
  scilab_rt_disp_s0_("a ~= c");
  int _tmp0[3][2][3];
  scilab_rt_ne_d3d3_i3(3,2,3,_u_a,3,2,3,_u_c,3,2,3,_tmp0);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp0);
  scilab_rt_disp_s0_("b ~= d");
  int _tmp1[3][2][3];
  scilab_rt_ne_d3d3_i3(3,2,3,_u_b,3,2,3,_u_d,3,2,3,_tmp1);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp1);
  scilab_rt_disp_s0_("a ~= b");
  int _tmp2[3][2][3];
  scilab_rt_ne_d3d3_i3(3,2,3,_u_a,3,2,3,_u_b,3,2,3,_tmp2);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp2);
  scilab_rt_disp_s0_("b ~= a");
  int _tmp3[3][2][3];
  scilab_rt_ne_d3d3_i3(3,2,3,_u_b,3,2,3,_u_a,3,2,3,_tmp3);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp3);
  scilab_rt_disp_s0_("30. ~= a");
  int _tmp4[3][2][3];
  scilab_rt_ne_d0d3_i3(30.,3,2,3,_u_a,3,2,3,_tmp4);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp4);
  scilab_rt_disp_s0_("a ~= 30.");
  int _tmp5[3][2][3];
  scilab_rt_ne_d3d0_i3(3,2,3,_u_a,30.,3,2,3,_tmp5);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp5);
  scilab_rt_disp_s0_("10 ~= b");
  int _tmp6[3][2][3];
  scilab_rt_ne_i0d3_i3(10,3,2,3,_u_b,3,2,3,_tmp6);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp6);
  scilab_rt_disp_s0_("b ~= 10");
  int _tmp7[3][2][3];
  scilab_rt_ne_d3i0_i3(3,2,3,_u_b,10,3,2,3,_tmp7);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp7);
  scilab_rt_disp_s0_("20. ~= b");
  int _tmp8[3][2][3];
  scilab_rt_ne_d0d3_i3(20.,3,2,3,_u_b,3,2,3,_tmp8);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp8);
  scilab_rt_disp_s0_("20 ~= a");
  int _tmp9[3][2][3];
  scilab_rt_ne_i0d3_i3(20,3,2,3,_u_a,3,2,3,_tmp9);
  scilab_rt_display_s0i3_("ans",3,2,3,_tmp9);

  scilab_rt_terminate();
}

