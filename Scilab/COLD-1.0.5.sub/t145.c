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

  /* testing getdate */
  int _u_a[1][10];
  scilab_rt_getdate__i2(1,10,_u_a);
  double _u_x = scilab_rt_getdate_s0_("s");
  int _u_b[1][10];
  scilab_rt_getdate_d0_i2(_u_x,1,10,_u_b);

  scilab_rt_terminate();
}

