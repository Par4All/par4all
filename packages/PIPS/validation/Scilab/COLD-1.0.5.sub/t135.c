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

  int _u_a2[2][3];
  _u_a2[0][0]=1;
  _u_a2[0][1]=0;
  _u_a2[0][2]=3;
  _u_a2[1][0]=4;
  _u_a2[1][1]=5;
  _u_a2[1][2]=6;
  scilab_rt_display_s0i2_("a2",2,3,_u_a2);
  double _u_b2[2][3];
  _u_b2[0][0]=4;
  _u_b2[0][1]=5;
  _u_b2[0][2]=6.;
  _u_b2[1][0]=0;
  _u_b2[1][1]=8;
  _u_b2[1][2]=9;
  scilab_rt_display_s0d2_("b2",2,3,_u_b2);
  int _tmp0[2][3];
  scilab_rt_and_i2d2_i2(2,3,_u_a2,2,3,_u_b2,2,3,_tmp0);
  scilab_rt_display_s0i2_("ans",2,3,_tmp0);

  scilab_rt_terminate();
}

