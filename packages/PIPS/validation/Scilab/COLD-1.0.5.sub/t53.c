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

  double complex _tmpxx0 = (4*I);
  double complex _u_z1 = (3+_tmpxx0);
  scilab_rt_display_s0z0_("z1",_u_z1);
  double complex _tmpxx1 = (4*I);
  double complex _u_z2 = (3-_tmpxx1);
  scilab_rt_display_s0z0_("z2",_u_z2);
  int _tmpxx2 = (-3);
  double complex _tmpxx3 = (4*I);
  double complex _u_z3 = (_tmpxx2+_tmpxx3);
  scilab_rt_display_s0z0_("z3",_u_z3);
  int _tmpxx4 = (-3);
  double complex _tmpxx5 = (4*I);
  double complex _u_z4 = (_tmpxx4-_tmpxx5);
  scilab_rt_display_s0z0_("z4",_u_z4);

  scilab_rt_terminate();
}

