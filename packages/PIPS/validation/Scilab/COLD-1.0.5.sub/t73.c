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

  /*  t73.sce: testing allocation with reals, rt */
  double _tmp0[3][3];
  scilab_rt_ones_i0i0_d2(3.5,3.5,3,3,_tmp0);
  scilab_rt_display_s0d2_("ans",3,3,_tmp0);
  double _tmp1[3][3];
  scilab_rt_eye_i0i0_d2(3.5,3.5,3,3,_tmp1);
  scilab_rt_display_s0d2_("ans",3,3,_tmp1);
  double _tmp2[3][3];
  scilab_rt_zeros_i0i0_d2(3.5,3.5,3,3,_tmp2);
  scilab_rt_display_s0d2_("ans",3,3,_tmp2);
  double _tmp3[3][3];
  scilab_rt_rand_i0i0_d2(3.5,3.5,3,3,_tmp3);
  scilab_rt_display_s0d2_("ans",3,3,_tmp3);

  scilab_rt_terminate();
}

