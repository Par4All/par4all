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

  double _u_x = scilab_rt_cos_d0_(1.);
  double _u_y = scilab_rt_cos_d0_(1.);
  int _u_k = 3;
  int _u_nmax = 10;
  double _tmpxx0 = (_u_x*_u_x);
  double _tmpxx1 = (_u_y*_u_y);
  double _tmpxx2 = (_tmpxx0+_tmpxx1);
  int _tmpxx3 = (_tmpxx2<4);
  int _tmpxx4 = (_u_k<_u_nmax);
  int _u_a = (_tmpxx3 && _tmpxx4);
  scilab_rt_display_s0i0_("a",_u_a);

  scilab_rt_terminate();
}

