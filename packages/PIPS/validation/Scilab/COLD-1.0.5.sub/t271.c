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

  /*  t271.sce _ size returning tuple */
  int _u_a1 = 1;
  int _u_a2[2][5];
  _u_a2[0][0]=1;
  _u_a2[0][1]=2;
  _u_a2[0][2]=3;
  _u_a2[0][3]=4;
  _u_a2[0][4]=5;
  _u_a2[1][0]=6;
  _u_a2[1][1]=7;
  _u_a2[1][2]=8;
  _u_a2[1][3]=9;
  _u_a2[1][4]=10;
  double _u_a3[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_a3);
  int _u_r1;
  int _u_c1;
  scilab_rt_size_i0_i0i0(_u_a1,&_u_r1,&_u_c1);
  scilab_rt_display_s0i0_("c1",_u_c1);
  scilab_rt_display_s0i0_("r1",_u_r1);
  int _u_r2;
  int _u_c2;
  scilab_rt_size_i2_i0i0(2,5,_u_a2,&_u_r2,&_u_c2);
  scilab_rt_display_s0i0_("c2",_u_c2);
  scilab_rt_display_s0i0_("r2",_u_r2);
  int _u_r3;
  int _u_c3;
  int _u_k3;
  scilab_rt_size_d3_i0i0i0(3,2,3,_u_a3,&_u_r3,&_u_c3,&_u_k3);
  scilab_rt_display_s0i0_("k3",_u_k3);
  scilab_rt_display_s0i0_("c3",_u_c3);
  scilab_rt_display_s0i0_("r3",_u_r3);
  int _u_s1[1][2];
  scilab_rt_size_i0_i2(_u_a1,1,2,_u_s1);
  scilab_rt_display_s0i2_("s1",1,2,_u_s1);
  int _u_s2[1][2];
  scilab_rt_size_i2_i2(2,5,_u_a2,1,2,_u_s2);
  scilab_rt_display_s0i2_("s2",1,2,_u_s2);
  int _u_s3[1][3];
  scilab_rt_size_d3_i2(3,2,3,_u_a3,1,3,_u_s3);
  scilab_rt_display_s0i2_("s3",1,3,_u_s3);

  scilab_rt_terminate();
}

