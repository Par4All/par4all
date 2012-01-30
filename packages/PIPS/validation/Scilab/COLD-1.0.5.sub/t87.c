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

  /*  t87.sce: testing mean function */
  int _u_a[3][3];
  _u_a[0][0]=1;
  _u_a[0][1]=12;
  _u_a[0][2]=3;
  _u_a[1][0]=4;
  _u_a[1][1]=5;
  _u_a[1][2]=86;
  _u_a[2][0]=17;
  _u_a[2][1]=8;
  _u_a[2][2]=29;
  double _u_b;
  scilab_rt_mean_i2_d0(3,3,_u_a,&_u_b);
  scilab_rt_display_s0d0_("b",_u_b);
  double _u_c[1][3];
  scilab_rt_mean_i2s0_d2(3,3,_u_a,"r",1,3,_u_c);
  scilab_rt_display_s0d2_("c",1,3,_u_c);
  double _u_d[3][1];
  scilab_rt_mean_i2s0_d2(3,3,_u_a,"c",3,1,_u_d);
  scilab_rt_display_s0d2_("d",3,1,_u_d);
  double _u_e;
  scilab_rt_mean_d2s0_d0(1,3,_u_c,"c",&_u_e);
  scilab_rt_display_s0d0_("e",_u_e);
  double _tmp0;
  scilab_rt_mean_d2s0_d0(3,1,_u_d,"r",&_tmp0);
  scilab_rt_display_s0d0_("ans",_tmp0);
  double _u_f[1][3];
  scilab_rt_mean_i2s0_d2(3,3,_u_a,"m",1,3,_u_f);
  scilab_rt_display_s0d2_("f",1,3,_u_f);
  double _u_g[1][3];
  scilab_rt_mean_i2i0_d2(3,3,_u_a,1,1,3,_u_g);
  scilab_rt_display_s0d2_("g",1,3,_u_g);
  double _u_h[3][1];
  scilab_rt_mean_i2i0_d2(3,3,_u_a,2,3,1,_u_h);
  scilab_rt_display_s0d2_("h",3,1,_u_h);
  double _u_ad[3][3];
  _u_ad[0][0]=1;
  _u_ad[0][1]=12.0;
  _u_ad[0][2]=3;
  _u_ad[1][0]=4;
  _u_ad[1][1]=5;
  _u_ad[1][2]=86;
  _u_ad[2][0]=17.0;
  _u_ad[2][1]=8;
  _u_ad[2][2]=29;
  double _u_fd[1][3];
  scilab_rt_mean_d2s0_d2(3,3,_u_ad,"m",1,3,_u_fd);
  scilab_rt_display_s0d2_("fd",1,3,_u_fd);
  double _u_gd[1][3];
  scilab_rt_mean_d2i0_d2(3,3,_u_ad,1,1,3,_u_gd);
  scilab_rt_display_s0d2_("gd",1,3,_u_gd);
  double _u_hd[3][1];
  scilab_rt_mean_d2i0_d2(3,3,_u_ad,2,3,1,_u_hd);
  scilab_rt_display_s0d2_("hd",3,1,_u_hd);
  int _u_ac[9][1];
  _u_ac[0][0]=1;
  _u_ac[1][0]=12;
  _u_ac[2][0]=3;
  _u_ac[3][0]=4;
  _u_ac[4][0]=5;
  _u_ac[5][0]=86;
  _u_ac[6][0]=17;
  _u_ac[7][0]=8;
  _u_ac[8][0]=29;
  int _u_ar[1][9];
  _u_ar[0][0]=1;
  _u_ar[0][1]=12;
  _u_ar[0][2]=3;
  _u_ar[0][3]=4;
  _u_ar[0][4]=5;
  _u_ar[0][5]=86;
  _u_ar[0][6]=17;
  _u_ar[0][7]=8;
  _u_ar[0][8]=29;
  double _u_r;
  scilab_rt_mean_i2s0_d0(1,9,_u_ar,"m",&_u_r);
  scilab_rt_display_s0d0_("r",_u_r);
  double _u_c2;
  scilab_rt_mean_i2s0_d0(9,1,_u_ac,"m",&_u_c2);
  scilab_rt_display_s0d0_("c2",_u_c2);
  double _u_adc[9][1];
  _u_adc[0][0]=1;
  _u_adc[1][0]=12.0;
  _u_adc[2][0]=3;
  _u_adc[3][0]=4;
  _u_adc[4][0]=5;
  _u_adc[5][0]=86;
  _u_adc[6][0]=17;
  _u_adc[7][0]=8;
  _u_adc[8][0]=29;
  double _u_adr[1][9];
  _u_adr[0][0]=1;
  _u_adr[0][1]=12;
  _u_adr[0][2]=3;
  _u_adr[0][3]=4;
  _u_adr[0][4]=5.0;
  _u_adr[0][5]=86;
  _u_adr[0][6]=17;
  _u_adr[0][7]=8;
  _u_adr[0][8]=29;
  double _u_rd;
  scilab_rt_mean_d2s0_d0(1,9,_u_adr,"m",&_u_rd);
  scilab_rt_display_s0d0_("rd",_u_rd);
  double _u_cd;
  scilab_rt_mean_d2s0_d0(9,1,_u_adc,"m",&_u_cd);
  scilab_rt_display_s0d0_("cd",_u_cd);
  int _tmp1[1][2];
  scilab_rt_size_d0_i2(_u_r,1,2,_tmp1);
  scilab_rt_display_s0i2_("ans",1,2,_tmp1);
  int _tmp2[1][2];
  scilab_rt_size_d0_i2(_u_c2,1,2,_tmp2);
  scilab_rt_display_s0i2_("ans",1,2,_tmp2);
  int _tmp3[1][2];
  scilab_rt_size_d0_i2(_u_rd,1,2,_tmp3);
  scilab_rt_display_s0i2_("ans",1,2,_tmp3);
  int _tmp4[1][2];
  scilab_rt_size_d0_i2(_u_cd,1,2,_tmp4);
  scilab_rt_display_s0i2_("ans",1,2,_tmp4);
  double _u_a2[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_a2);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_a2[1][j][k] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_a2[i][1][k] = 20;
    }
  }

  for(int i=0; i<3;++i) {
    _u_a2[i][1][2] = 30;
  }
  scilab_rt_display_s0d3_("a2",3,2,3,_u_a2);
  double _u_t1[1][2][3];
  scilab_rt_mean_d3s0_d3(3,2,3,_u_a2,"r",1,2,3,_u_t1);
  scilab_rt_display_s0d3_("t1",1,2,3,_u_t1);
  double _u_t2[3][1][3];
  scilab_rt_mean_d3s0_d3(3,2,3,_u_a2,"c",3,1,3,_u_t2);
  scilab_rt_display_s0d3_("t2",3,1,3,_u_t2);
  double _u_t3[1][2][3];
  scilab_rt_mean_d3s0_d3(3,2,3,_u_a2,"m",1,2,3,_u_t3);
  scilab_rt_display_s0d3_("t3",1,2,3,_u_t3);
  double _u_t4[1][2][3];
  scilab_rt_mean_d3i0_d3(3,2,3,_u_a2,1,1,2,3,_u_t4);
  scilab_rt_display_s0d3_("t4",1,2,3,_u_t4);
  double _u_t5[3][1][3];
  scilab_rt_mean_d3i0_d3(3,2,3,_u_a2,2,3,1,3,_u_t5);
  scilab_rt_display_s0d3_("t5",3,1,3,_u_t5);
  double _u_t6[3][2][1];
  scilab_rt_mean_d3i0_d3(3,2,3,_u_a2,3,3,2,1,_u_t6);
  scilab_rt_display_s0d3_("t6",3,2,1,_u_t6);

  scilab_rt_terminate();
}

