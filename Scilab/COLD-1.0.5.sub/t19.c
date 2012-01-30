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

  int _tmpxx0[1][2];
  _tmpxx0[0][0]=1;
  _tmpxx0[0][1]=2;
  int _u_a[1][2];
  scilab_rt_sub_i2_i2(1,2,_tmpxx0,1,2,_u_a);
  scilab_rt_display_s0i2_("a",1,2,_u_a);
  int _tmpxx1[1][2];
  _tmpxx1[0][0]=1;
  _tmpxx1[0][1]=2;
  int _u_b[1][2];
  scilab_rt_sub_i2_i2(1,2,_tmpxx1,1,2,_u_b);
  scilab_rt_display_s0i2_("b",1,2,_u_b);
  int _tmpxx2[1][2];
  _tmpxx2[0][0]=1;
  _tmpxx2[0][1]=2;
  int _u_c[1][2];
  scilab_rt_sub_i2_i2(1,2,_tmpxx2,1,2,_u_c);
  scilab_rt_display_s0i2_("c",1,2,_u_c);
  int _tmpxx3[1][2];
  _tmpxx3[0][0]=1;
  _tmpxx3[0][1]=2;
  int _u_d[1][2];
  scilab_rt_sub_i2_i2(1,2,_tmpxx3,1,2,_u_d);
  scilab_rt_display_s0i2_("d",1,2,_u_d);
  double _tmpxx4 = scilab_rt_cos_d0_((SCILAB_PI / 4.));
  double _u_e = (-_tmpxx4);
  scilab_rt_display_s0d0_("e",_u_e);
  double _u_aa[10][10];
  scilab_rt_ones_i0i0_d2(10,10,10,10,_u_aa);
  double _tmpxx5 = _u_aa[0][0];
  double _tmpxx6 = _u_aa[0][1];
  double _tmpxx7 = (-_tmpxx5);
  double _tmpxx8 = (_tmpxx6*2);
  double _u_f = (_tmpxx7-_tmpxx8);
  scilab_rt_display_s0d0_("f",_u_f);

  scilab_rt_terminate();
}

