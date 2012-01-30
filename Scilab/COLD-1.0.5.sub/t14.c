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

  double _u_nu = 0.1;
  double _tmpxx0 = (3*_u_nu);
  double _tmpxx1 = (1. / 2.);
  double _tmpxx2 = (_tmpxx0+_tmpxx1);
  double _u_omega = (1. / _tmpxx2);
  if ((_u_omega!=1.25)) {
    scilab_rt_disp_s0_("t14 FAILED");
  } else { 
    scilab_rt_disp_s0_("T14 PASSED");
  }

  scilab_rt_terminate();
}

