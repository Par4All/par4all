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

  /*  parser test => 1 */
  double _tmpxx0 = (2.*2.);
  double _tmpxx1 = (1. / _tmpxx0);
  double _tmp0 = (_tmpxx1*4.);
  scilab_rt_display_s0d0_("ans",_tmp0);

  scilab_rt_terminate();
}

