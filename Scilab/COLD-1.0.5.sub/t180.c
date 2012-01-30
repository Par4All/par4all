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

  int _u_a[3][3];
  _u_a[0][0]=0;
  _u_a[0][1]=0;
  _u_a[0][2]=0;
  _u_a[1][0]=1;
  _u_a[1][1]=1;
  _u_a[1][2]=1;
  _u_a[2][0]=0;
  _u_a[2][1]=0;
  _u_a[2][2]=0;
  scilab_rt_display_s0i2_("a",3,3,_u_a);
  int _u_aBool[3][3];
  scilab_rt_bool2s_i2_i2(3,3,_u_a,3,3,_u_aBool);
  scilab_rt_display_s0i2_("aBool",3,3,_u_aBool);
  int _u_b[3][3];
  _u_b[0][0]=0;
  _u_b[0][1]=2;
  _u_b[0][2]=3;
  _u_b[1][0]=4;
  _u_b[1][1]=0;
  _u_b[1][2]=6;
  _u_b[2][0]=7;
  _u_b[2][1]=8;
  _u_b[2][2]=0;
  scilab_rt_display_s0i2_("b",3,3,_u_b);
  int _u_bBool[3][3];
  scilab_rt_bool2s_i2_i2(3,3,_u_b,3,3,_u_bBool);
  scilab_rt_display_s0i2_("bBool",3,3,_u_bBool);
  double _u_c[3][3];
  _u_c[0][0]=0;
  _u_c[0][1]=2;
  _u_c[0][2]=3.2;
  _u_c[1][0]=4;
  _u_c[1][1]=0.0;
  _u_c[1][2]=6;
  _u_c[2][0]=7;
  _u_c[2][1]=8.9;
  _u_c[2][2]=0;
  scilab_rt_display_s0d2_("c",3,3,_u_c);
  int _u_cBool[3][3];
  scilab_rt_bool2s_d2_i2(3,3,_u_c,3,3,_u_cBool);
  scilab_rt_display_s0i2_("cBool",3,3,_u_cBool);
  int _u_d[3][4];
  _u_d[0][0]=0;
  _u_d[0][1]=2;
  _u_d[0][2]=3;
  _u_d[0][3]=3;
  _u_d[1][0]=4;
  _u_d[1][1]=0;
  _u_d[1][2]=6;
  _u_d[1][3]=6;
  _u_d[2][0]=7;
  _u_d[2][1]=8;
  _u_d[2][2]=0;
  _u_d[2][3]=0;
  scilab_rt_display_s0i2_("d",3,4,_u_d);
  int _u_dBool[3][4];
  scilab_rt_bool2s_i2_i2(3,4,_u_d,3,4,_u_dBool);
  scilab_rt_display_s0i2_("dBool",3,4,_u_dBool);

  scilab_rt_terminate();
}

