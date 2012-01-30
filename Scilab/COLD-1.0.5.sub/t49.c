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

  int _tmpxx0[1][2];
  _tmpxx0[0][0]=1;
  _tmpxx0[0][1]=2;
  double _u_a[1][2];
  scilab_rt_cos_i2_d2(1,2,_tmpxx0,1,2,_u_a);
  scilab_rt_display_s0d2_("a",1,2,_u_a);

  scilab_rt_terminate();
}

