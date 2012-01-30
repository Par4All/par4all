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

  int _u_a[1][4];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[0][3]=4;
  double _u_an;
  scilab_rt_norm_i2_d0(1,4,_u_a,&_u_an);
  scilab_rt_display_s0d0_("an",_u_an);
  double _u_b[1][4];
  _u_b[0][0]=1.0;
  _u_b[0][1]=2;
  _u_b[0][2]=3;
  _u_b[0][3]=4;
  double _u_bn;
  scilab_rt_norm_d2_d0(1,4,_u_b,&_u_bn);
  scilab_rt_display_s0d0_("bn",_u_bn);
  double _u_c[6][1];
  _u_c[0][0]=5.2;
  _u_c[1][0]=64.8;
  _u_c[2][0]=2.3;
  _u_c[3][0]=1.4;
  _u_c[4][0]=5.6;
  _u_c[5][0]=6.2;
  double _u_cn;
  scilab_rt_norm_d2_d0(6,1,_u_c,&_u_cn);
  scilab_rt_display_s0d0_("cn",_u_cn);

  scilab_rt_terminate();
}

