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

  /*  t123.sce: Test 3D matrices multiplication, division and elementwise multiplication and division */
  double _u_a[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_a);
  double _u_b[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_b);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_b[1][j][k] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_b[i][1][k] = 20;
    }
  }

  for(int i=0; i<3;++i) {
    _u_b[i][1][2] = 30;
  }
  int _u_c = 2;
  double _u_d = 1.;
  scilab_rt_disp_s0_("multiplication by a scalar");
  double _tmp0[3][2][3];
  scilab_rt_mul_d3i0_d3(3,2,3,_u_a,_u_c,3,2,3,_tmp0);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp0);
  double _tmp1[3][2][3];
  scilab_rt_mul_i0d3_d3(_u_c,3,2,3,_u_a,3,2,3,_tmp1);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp1);
  double _tmp2[3][2][3];
  scilab_rt_mul_d3d0_d3(3,2,3,_u_a,_u_d,3,2,3,_tmp2);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp2);
  double _tmp3[3][2][3];
  scilab_rt_mul_d0d3_d3(_u_d,3,2,3,_u_a,3,2,3,_tmp3);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp3);
  double _tmp4[3][2][3];
  scilab_rt_mul_d3i0_d3(3,2,3,_u_b,_u_c,3,2,3,_tmp4);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp4);
  double _tmp5[3][2][3];
  scilab_rt_mul_i0d3_d3(_u_c,3,2,3,_u_b,3,2,3,_tmp5);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp5);
  double _tmp6[3][2][3];
  scilab_rt_mul_d3d0_d3(3,2,3,_u_b,_u_d,3,2,3,_tmp6);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp6);
  double _tmp7[3][2][3];
  scilab_rt_mul_d0d3_d3(_u_d,3,2,3,_u_b,3,2,3,_tmp7);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp7);
  scilab_rt_disp_s0_("division by a scalar");
  double _tmp8[3][2][3];
  scilab_rt_div_d3i0_d3(3,2,3,_u_a,_u_c,3,2,3,_tmp8);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp8);
  double _tmp9[3][2][3];
  scilab_rt_div_d3d0_d3(3,2,3,_u_a,_u_d,3,2,3,_tmp9);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp9);
  double _tmp10[3][2][3];
  scilab_rt_div_d3i0_d3(3,2,3,_u_b,_u_c,3,2,3,_tmp10);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp10);
  double _tmp11[3][2][3];
  scilab_rt_div_d3d0_d3(3,2,3,_u_b,_u_d,3,2,3,_tmp11);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp11);
  scilab_rt_disp_s0_("elementwise multiplication by a scalar");
  double _tmp12[3][2][3];
  scilab_rt_eltmul_d3i0_d3(3,2,3,_u_a,_u_c,3,2,3,_tmp12);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp12);
  double _tmp13[3][2][3];
  scilab_rt_eltmul_i0d3_d3(_u_c,3,2,3,_u_a,3,2,3,_tmp13);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp13);
  double _tmp14[3][2][3];
  scilab_rt_eltmul_d3d0_d3(3,2,3,_u_a,_u_d,3,2,3,_tmp14);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp14);
  double _tmp15[3][2][3];
  scilab_rt_eltmul_d0d3_d3(_u_d,3,2,3,_u_a,3,2,3,_tmp15);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp15);
  double _tmp16[3][2][3];
  scilab_rt_eltmul_d3i0_d3(3,2,3,_u_b,_u_c,3,2,3,_tmp16);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp16);
  double _tmp17[3][2][3];
  scilab_rt_eltmul_i0d3_d3(_u_c,3,2,3,_u_b,3,2,3,_tmp17);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp17);
  double _tmp18[3][2][3];
  scilab_rt_eltmul_d3d0_d3(3,2,3,_u_b,_u_d,3,2,3,_tmp18);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp18);
  double _tmp19[3][2][3];
  scilab_rt_eltmul_d0d3_d3(_u_d,3,2,3,_u_b,3,2,3,_tmp19);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp19);
  scilab_rt_disp_s0_("elementwise division by a scalar");
  double _tmp20[3][2][3];
  scilab_rt_eltdiv_d3i0_d3(3,2,3,_u_a,_u_c,3,2,3,_tmp20);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp20);
  double _tmp21[3][2][3];
  scilab_rt_eltdiv_d3d0_d3(3,2,3,_u_a,_u_d,3,2,3,_tmp21);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp21);
  double _tmp22[3][2][3];
  scilab_rt_eltdiv_d3i0_d3(3,2,3,_u_b,_u_c,3,2,3,_tmp22);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp22);
  double _tmp23[3][2][3];
  scilab_rt_eltdiv_d3d0_d3(3,2,3,_u_b,_u_d,3,2,3,_tmp23);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp23);

  scilab_rt_terminate();
}

