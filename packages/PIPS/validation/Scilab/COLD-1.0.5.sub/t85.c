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

  double _tmpc0[1][2];
  scilab_rt_ones_i0i0_d2(1,2,1,2,_tmpc0);
  double _tmpc1[2][1];
  scilab_rt_ones_i0i0_d2(2,1,2,1,_tmpc1);

  scilab_rt_terminate();
}

