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

  int _u_a = 1;
  int _u_b = 2;
  int _u_c = (_u_a+_u_b);
  scilab_rt_display_s0i0_("c",_u_c);

  scilab_rt_terminate();
}

