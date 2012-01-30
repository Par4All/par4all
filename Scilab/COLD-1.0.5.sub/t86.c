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
  _u_a[0][0]=1;
  _u_a[0][1]=12;
  _u_a[0][2]=3;
  _u_a[1][0]=4;
  _u_a[1][1]=5;
  _u_a[1][2]=6;
  _u_a[2][0]=17;
  _u_a[2][1]=8;
  _u_a[2][2]=9;
  int _u_b[3][1];
  scilab_rt_max_i2s0_i2(3,3,_u_a,"c",3,1,_u_b);
  scilab_rt_display_s0i2_("b",3,1,_u_b);
  int _u_c[1][3];
  scilab_rt_max_i2s0_i2(3,3,_u_a,"r",1,3,_u_c);
  scilab_rt_display_s0i2_("c",1,3,_u_c);
  int _u_d[3][1];
  scilab_rt_max_i2s0_i2(3,1,_u_b,"c",3,1,_u_d);
  scilab_rt_display_s0i2_("d",3,1,_u_d);
  int _u_e;
  scilab_rt_max_i2s0_i0(3,1,_u_b,"r",&_u_e);
  scilab_rt_display_s0i0_("e",_u_e);
  int _u_b2[3][1];
  scilab_rt_min_i2s0_i2(3,3,_u_a,"c",3,1,_u_b2);
  scilab_rt_display_s0i2_("b2",3,1,_u_b2);
  int _u_c2[1][3];
  scilab_rt_min_i2s0_i2(3,3,_u_a,"r",1,3,_u_c2);
  scilab_rt_display_s0i2_("c2",1,3,_u_c2);
  int _u_d2[3][1];
  scilab_rt_min_i2s0_i2(3,1,_u_b2,"c",3,1,_u_d2);
  scilab_rt_display_s0i2_("d2",3,1,_u_d2);
  int _u_e2;
  scilab_rt_min_i2s0_i0(3,1,_u_b2,"r",&_u_e2);
  scilab_rt_display_s0i0_("e2",_u_e2);
  double _u_l[3][3];
  _u_l[0][0]=1.0;
  _u_l[0][1]=12;
  _u_l[0][2]=3;
  _u_l[1][0]=4;
  _u_l[1][1]=5;
  _u_l[1][2]=6;
  _u_l[2][0]=17;
  _u_l[2][1]=8;
  _u_l[2][2]=9;
  scilab_rt_display_s0d2_("l",3,3,_u_l);
  int _u_n = 6;
  double _u_p1[3][3];
  scilab_rt_max_d2d0_d2(3,3,_u_l,_u_n,3,3,_u_p1);
  scilab_rt_display_s0d2_("p1",3,3,_u_p1);
  double _u_o1[3][3];
  scilab_rt_max_d0d2_d2((_u_n-1),3,3,_u_l,3,3,_u_o1);
  scilab_rt_display_s0d2_("o1",3,3,_u_o1);
  double _u_p2[3][3];
  scilab_rt_max_d2d0_d2(3,3,_u_l,_u_n,3,3,_u_p2);
  scilab_rt_display_s0d2_("p2",3,3,_u_p2);
  double _u_o2[3][3];
  scilab_rt_max_d0d2_d2((_u_n-1),3,3,_u_l,3,3,_u_o2);
  scilab_rt_display_s0d2_("o2",3,3,_u_o2);

  scilab_rt_terminate();
}

