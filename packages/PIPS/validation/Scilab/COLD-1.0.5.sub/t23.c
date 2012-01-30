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

  double _tmp0[10][10];
  scilab_rt_ones_i0i0_d2(10,10,10,10,_tmp0);
  scilab_rt_display_s0d2_("ans",10,10,_tmp0);

  scilab_rt_terminate();
}

