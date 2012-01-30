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

  /*  t136.sce: testing or function */
  int _u_a2[3][3];
  _u_a2[0][0]=1;
  _u_a2[0][1]=2;
  _u_a2[0][2]=3;
  _u_a2[1][0]=4;
  _u_a2[1][1]=5;
  _u_a2[1][2]=6;
  _u_a2[2][0]=7;
  _u_a2[2][1]=8;
  _u_a2[2][2]=9;
  int _u_b;
  scilab_rt_or_i2_i0(3,3,_u_a2,&_u_b);
  scilab_rt_display_s0i0_("b",_u_b);
  int _u_c[1][3];
  scilab_rt_or_i2s0_i2(3,3,_u_a2,"r",1,3,_u_c);
  scilab_rt_display_s0i2_("c",1,3,_u_c);
  _u_a2[2][1] = 0;
  int _u_d;
  scilab_rt_or_i2_i0(3,3,_u_a2,&_u_d);
  scilab_rt_display_s0i0_("d",_u_d);
  double _u_a3[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_a3);
  scilab_rt_display_s0d3_("a3",3,2,3,_u_a3);

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
  int _u_e;
  scilab_rt_or_d3_i0(3,2,3,_u_a3,&_u_e);
  scilab_rt_display_s0i0_("e",_u_e);
  _u_a3[0][0][1] = 0;
  int _u_f;
  scilab_rt_or_d3_i0(3,2,3,_u_a3,&_u_f);
  scilab_rt_display_s0i0_("f",_u_f);
  double _u_ad[2][3];
  _u_ad[0][0]=1;
  _u_ad[0][1]=2;
  _u_ad[0][2]=3.4;
  _u_ad[1][0]=4;
  _u_ad[1][1]=5;
  _u_ad[1][2]=6.2;
  int _u_g;
  scilab_rt_or_d2_i0(2,3,_u_ad,&_u_g);
  scilab_rt_display_s0i0_("g",_u_g);
  double complex _tmpxx0 = (3*I);
  double complex _tmpxx1 = (1*I);
  double complex _tmpxx2 = (2*I);
  double complex _tmpxx3 = (2*I);
  double complex _u_ac[2][2];
  _u_ac[0][0]=(2+_tmpxx0);
  _u_ac[0][1]=(3+_tmpxx1);
  _u_ac[1][0]=(5+_tmpxx2);
  _u_ac[1][1]=(4+_tmpxx3);
  scilab_rt_display_s0z2_("ac",2,2,_u_ac);
  int _u_h;
  scilab_rt_or_z2_i0(2,2,_u_ac,&_u_h);
  scilab_rt_display_s0i0_("h",_u_h);
  double _u_z2[2][3];
  scilab_rt_zeros_i0i0_d2(2,3,2,3,_u_z2);
  double _u_z3[2][3][2];
  scilab_rt_zeros_i0i0i0_d3(2,3,2,2,3,2,_u_z3);
  int _u_i;
  scilab_rt_or_d2_i0(2,3,_u_z2,&_u_i);
  scilab_rt_display_s0i0_("i",_u_i);
  int _u_j;
  scilab_rt_or_d3_i0(2,3,2,_u_z3,&_u_j);
  scilab_rt_display_s0i0_("j",_u_j);
  double _u_o2[2][3];
  scilab_rt_ones_i0i0_d2(2,3,2,3,_u_o2);
  double _u_o3[2][3][2];
  scilab_rt_ones_i0i0i0_d3(2,3,2,2,3,2,_u_o3);
  int _u_k;
  scilab_rt_or_d2_i0(2,3,_u_o2,&_u_k);
  scilab_rt_display_s0i0_("k",_u_k);
  int _u_l;
  scilab_rt_or_d3_i0(2,3,2,_u_o3,&_u_l);
  scilab_rt_display_s0i0_("l",_u_l);

  scilab_rt_terminate();
}

