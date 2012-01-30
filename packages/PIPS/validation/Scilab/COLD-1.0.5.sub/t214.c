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

  /*  t214.sce _ spec function */
  int _u_a[3][3];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[1][0]=4;
  _u_a[1][1]=5;
  _u_a[1][2]=6;
  _u_a[2][0]=7;
  _u_a[2][1]=8;
  _u_a[2][2]=9;
  scilab_rt_display_s0i2_("a",3,3,_u_a);
  double _u_Y[3][3];
  double _u_D[3][3];
  scilab_rt_spec_i2_d2d2(3,3,_u_a,3,3,_u_Y,3,3,_u_D);
  scilab_rt_display_s0d2_("D",3,3,_u_D);
  scilab_rt_display_s0d2_("Y",3,3,_u_Y);
  int _u_b[3][3];
  _u_b[0][0]=1;
  _u_b[0][1]=2;
  _u_b[0][2]=3;
  _u_b[1][0]=2;
  _u_b[1][1]=11;
  _u_b[1][2]=6;
  _u_b[2][0]=3;
  _u_b[2][1]=6;
  _u_b[2][2]=10;
  scilab_rt_display_s0i2_("b",3,3,_u_b);
  /*  Sym */
  
  
  scilab_rt_spec_i2_d2d2(3,3,_u_b,3,3,_u_Y,3,3,_u_D);
  scilab_rt_display_s0d2_("D",3,3,_u_D);
  scilab_rt_display_s0d2_("Y",3,3,_u_Y);

  scilab_rt_terminate();
}

