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

  double _u_a[10][10];
  scilab_rt_ones_i0i0_d2(10,10,10,10,_u_a);
  for (int _u_i=1; _u_i<=10; _u_i++) {
    _u_a[0][_u_i-1] = _u_i;
  }
  double _tmpxx0 = _u_a[0][1];
  double _tmpxx1 = _u_a[0][2];
  double _tmpxx2 = _u_a[0][0];
  double _tmpxx3 = (_tmpxx0*_tmpxx1);
  double _u_b = (_tmpxx2+_tmpxx3);
  scilab_rt_display_s0d0_("b",_u_b);
  double _tmpxx4 = _u_a[0][0];
  double _tmpxx5 = _u_a[0][1];
  double _tmpxx6 = (_tmpxx4*_tmpxx5);
  double _tmpxx7 = _u_a[0][2];
  double _u_c = (_tmpxx6+_tmpxx7);
  scilab_rt_display_s0d0_("c",_u_c);

  scilab_rt_terminate();
}

