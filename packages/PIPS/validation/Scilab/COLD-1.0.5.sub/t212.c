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

  /*  t212.sce _ inv function */
  double _u_a[3][3];
  _u_a[0][0]=2.0;
  _u_a[0][1]=4;
  _u_a[0][2]=0;
  _u_a[1][0]=6;
  _u_a[1][1]=2;
  _u_a[1][2]=(-1);
  _u_a[2][0]=0;
  _u_a[2][1]=(-8);
  _u_a[2][2]=2;
  scilab_rt_display_s0d2_("a",3,3,_u_a);
  double _tmp0[3][3];
  scilab_rt_inv_d2_d2(3,3,_u_a,3,3,_tmp0);
  scilab_rt_display_s0d2_("ans",3,3,_tmp0);
  int _u_b[3][3];
  _u_b[0][0]=2;
  _u_b[0][1]=4;
  _u_b[0][2]=7;
  _u_b[1][0]=6;
  _u_b[1][1]=1;
  _u_b[1][2]=(-1);
  _u_b[2][0]=6;
  _u_b[2][1]=(-8);
  _u_b[2][2]=2;
  scilab_rt_display_s0i2_("b",3,3,_u_b);
  double _tmp1[3][3];
  scilab_rt_inv_i2_d2(3,3,_u_b,3,3,_tmp1);
  scilab_rt_display_s0d2_("ans",3,3,_tmp1);

  scilab_rt_terminate();
}

