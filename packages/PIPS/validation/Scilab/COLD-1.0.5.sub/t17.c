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

  double _tmpxx0 = scilab_rt_cos_i0_(2);
  double _tmpxx1 = scilab_rt_cos_i0_(3);
  double _tmpxx2 = scilab_rt_cos_i0_(1);
  double _tmpxx3 = (_tmpxx0*_tmpxx1);
  double _u_a = (_tmpxx2+_tmpxx3);
  scilab_rt_display_s0d0_("a",_u_a);
  double _tmpxx4 = scilab_rt_cos_i0_(10);
  double _tmpxx5 = scilab_rt_cos_i0_(11);
  double _tmpxx6 = (_tmpxx4*_tmpxx5);
  double _tmpxx7 = scilab_rt_cos_i0_(12);
  double _u_b = (_tmpxx6+_tmpxx7);
  scilab_rt_display_s0d0_("b",_u_b);

  scilab_rt_terminate();
}

