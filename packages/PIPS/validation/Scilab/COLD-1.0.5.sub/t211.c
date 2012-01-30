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

  /* t210.sce _ chol function */
  double _u_a[3][3];
  _u_a[0][0]=2.0;
  _u_a[0][1]=(-1);
  _u_a[0][2]=0;
  _u_a[1][0]=(-1);
  _u_a[1][1]=2;
  _u_a[1][2]=(-1);
  _u_a[2][0]=0;
  _u_a[2][1]=(-1);
  _u_a[2][2]=2;
  scilab_rt_display_s0d2_("a",3,3,_u_a);
  double _tmp0[3][3];
  scilab_rt_chol_d2_d2(3,3,_u_a,3,3,_tmp0);
  scilab_rt_display_s0d2_("ans",3,3,_tmp0);
  scilab_rt_display_s0d2_("a",3,3,_u_a);
  int _u_b[3][3];
  _u_b[0][0]=2;
  _u_b[0][1]=(-2);
  _u_b[0][2]=0;
  _u_b[1][0]=(-1);
  _u_b[1][1]=3;
  _u_b[1][2]=(-1);
  _u_b[2][0]=0;
  _u_b[2][1]=(-1);
  _u_b[2][2]=3;
  scilab_rt_display_s0i2_("b",3,3,_u_b);
  double _tmp1[3][3];
  scilab_rt_chol_i2_d2(3,3,_u_b,3,3,_tmp1);
  scilab_rt_display_s0d2_("ans",3,3,_tmp1);
  scilab_rt_display_s0i2_("b",3,3,_u_b);

  scilab_rt_terminate();
}

